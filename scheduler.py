"""
factory_scheduler.py
---------------------
Factory production scheduling prototype using OR-Tools CP-SAT.

Every order is OPTIONAL to schedule (never forced), so the solver never
returns INFEASIBLE. It schedules as many orders as it can - prioritizing
high-priority orders - and for anything it can't schedule, it explains why.
"""

import json
from pathlib import Path

from ortools.sat.python import cp_model


DATA_PATH = Path(__file__).parent / "data" / "factory_data.json"
BASE_HOUR = 8  # 08:00 == hour 0 in our internal representation


def load_data(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_hour_offset(time_str: str) -> float:
    hh, mm = time_str.split(":")
    return int(hh) - BASE_HOUR + (int(mm) / 60)


def build_and_solve(data: dict):
    workers = data["workers"]
    machines = data["machines"]
    orders = data["orders"]
    cfg = data["constraints"]

    horizon = 0
    for w in workers:
        horizon = max(horizon, int(to_hour_offset(w["shift"][1])))
    for m in machines:
        horizon = max(horizon, int(to_hour_offset(m["available"][1])))

    model = cp_model.CpModel()

    # -----------------------------------------------------------------
    # Decision variables
    # -----------------------------------------------------------------
    # presence[o,m,w]  : bool -> order o assigned to machine m + worker w
    # start[o,m,w]     : int  -> start hour, if that combo is chosen
    # scheduled[o]     : bool -> is order o scheduled AT ALL (on any combo)?
    #                            This is the key addition: it's the "opt-in"
    #                            variable that lets an order go unscheduled
    #                            instead of forcing infeasibility.
    #
    # A (machine, worker) combo only exists for an order if the machine has
    # the right capability AND the order's duration fits inside the overlap
    # of the machine's and worker's availability windows ("no overtime").
    presence = {}
    start = {}
    combos_by_order = {o["id"]: [] for o in orders}
    combos_by_machine = {m["id"]: [] for m in machines}
    combos_by_worker = {w["id"]: [] for w in workers}

    for o in orders:
        duration = o["duration_hours"]
        for m in machines:
            if o["product"] not in m["capabilities"]:
                continue

            m_start = to_hour_offset(m["available"][0])
            m_end = to_hour_offset(m["available"][1])

            for w in workers:
                w_start = to_hour_offset(w["shift"][0])
                w_end = to_hour_offset(w["shift"][1])

                earliest_start = max(m_start, w_start)
                latest_start = min(m_end, w_end) - duration

                if latest_start < earliest_start:
                    continue

                key = (o["id"], m["id"], w["id"])
                presence[key] = model.new_bool_var(f"presence_{o['id']}_{m['id']}_{w['id']}")
                start[key] = model.new_int_var(
                    int(earliest_start), int(latest_start), f"start_{o['id']}_{m['id']}_{w['id']}"
                )

                combos_by_order[o["id"]].append(key)
                combos_by_machine[m["id"]].append(key)
                combos_by_worker[w["id"]].append(key)

    # -----------------------------------------------------------------
    # CHANGE 1: orders are now OPTIONAL, not mandatory.
    #
    # Previously: model.add_exactly_one(...) forced every order onto some
    # combo, which is what made the model INFEASIBLE whenever capacity or
    # compatibility ran out.
    #
    # Now: model.add_at_most_one(...) allows an order to select zero combos
    # (i.e. go unscheduled), and a `scheduled[o]` bool tracks whether it did.
    # This is the proper OR-Tools pattern for optional tasks.
    # -----------------------------------------------------------------
    scheduled = {}
    for o in orders:
        candidates = combos_by_order[o["id"]]

        if not candidates:
            # No feasible (machine, worker) combo exists at all for this
            # order - it can never be scheduled, regardless of contention
            # with other orders. Fix scheduled=0 (represented as plain 0,
            # no variable needed).
            scheduled[o["id"]] = 0
            continue

        model.add_at_most_one(presence[key] for key in candidates)

        sched_var = model.new_bool_var(f"scheduled_{o['id']}")
        model.add(sched_var == sum(presence[key] for key in candidates))
        scheduled[o["id"]] = sched_var

    order_duration = {o["id"]: o["duration_hours"] for o in orders}

    # Machine / worker no-overlap constraints are unchanged - they still
    # use OptionalIntervalVar gated on `presence`, so combos that aren't
    # chosen simply don't occupy the timeline.
    if cfg.get("machine_can_process_only_one_order_at_a_time", True):
        for m in machines:
            intervals = []
            for key in combos_by_machine[m["id"]]:
                o_id, m_id, w_id = key
                duration = order_duration[o_id]
                intervals.append(
                    model.new_optional_interval_var(
                        start[key], duration, start[key] + duration, presence[key],
                        f"interval_m_{o_id}_{m_id}_{w_id}",
                    )
                )
            model.add_no_overlap(intervals)

    if cfg.get("worker_can_do_only_one_order_at_a_time", True):
        for w in workers:
            intervals = []
            for key in combos_by_worker[w["id"]]:
                o_id, m_id, w_id = key
                duration = order_duration[o_id]
                intervals.append(
                    model.new_optional_interval_var(
                        start[key], duration, start[key] + duration, presence[key],
                        f"interval_w_{o_id}_{m_id}_{w_id}",
                    )
                )
            model.add_no_overlap(intervals)

    # -----------------------------------------------------------------
    # CHANGE 2: objective now has FOUR tiers, matching the requested
    # priority order. Each tier's weight is large enough to completely
    # dominate every lower tier combined, so the solver satisfies them
    # in strict order (a lexicographic objective via weighted sum):
    #
    #   Tier 1 (highest): maximize orders scheduled, weighted by priority
    #                      -> leaving a priority-3 order unscheduled costs
    #                         far more than leaving a priority-1 order
    #                         unscheduled.
    #   Tier 2: minimize the COUNT of late orders
    #   Tier 3: minimize total priority-weighted lateness (hours)
    #   Tier 4: minimize total finish time (idle-time tiebreaker)
    # -----------------------------------------------------------------
    UNSCHEDULED_WEIGHT = 10_000_000
    LATE_COUNT_WEIGHT = 100_000
    LATENESS_WEIGHT = 1_000
    IDLE_WEIGHT = 1

    deadlines = {o["id"]: to_hour_offset(o["deadline"]) for o in orders}
    priorities = {o["id"]: o["priority"] for o in orders}

    unscheduled_penalty_terms = []
    late_count_terms = []
    lateness_terms = []
    finish_terms = []

    for o in orders:
        o_id = o["id"]
        priority = priorities[o_id]
        sched_var = scheduled[o_id]
        # Cost of NOT scheduling this order = priority * (1 - scheduled).
        # If sched_var is the constant 0 (structurally impossible order),
        # this just adds a fixed penalty every solution pays equally, so
        # it doesn't distort the search - it's a constant, not a variable.
        unscheduled_penalty_terms.append(priority * (1 - sched_var))

    for key, presence_var in presence.items():
        o_id, m_id, w_id = key
        duration = order_duration[o_id]
        deadline = deadlines[o_id]
        priority = priorities[o_id]

        lateness = model.new_int_var(0, horizon, f"lateness_{o_id}_{m_id}_{w_id}")
        model.add(lateness >= start[key] + duration - int(deadline))

        is_late = model.new_bool_var(f"is_late_{o_id}_{m_id}_{w_id}")
        model.add(lateness >= 1).only_enforce_if(is_late)
        model.add(lateness == 0).only_enforce_if(is_late.negated())

        # Gate both the lateness penalty and the late-count flag on
        # `presence`, so combos that weren't chosen contribute nothing.
        lateness_term = model.new_int_var(0, priority * horizon, f"late_term_{o_id}_{m_id}_{w_id}")
        model.add(lateness_term == priority * lateness).only_enforce_if(presence_var)
        model.add(lateness_term == 0).only_enforce_if(presence_var.negated())
        lateness_terms.append(lateness_term)

        late_count_term = model.new_bool_var(f"late_count_{o_id}_{m_id}_{w_id}")
        model.add(late_count_term == 1).only_enforce_if([presence_var, is_late])
        model.add(late_count_term == 0).only_enforce_if(presence_var.negated())
        model.add(late_count_term == 0).only_enforce_if(is_late.negated())
        late_count_terms.append(late_count_term)

        finish_term = model.new_int_var(0, horizon, f"finish_term_{o_id}_{m_id}_{w_id}")
        model.add(finish_term == start[key] + duration).only_enforce_if(presence_var)
        model.add(finish_term == 0).only_enforce_if(presence_var.negated())
        finish_terms.append(finish_term)

    model.minimize(
        UNSCHEDULED_WEIGHT * sum(unscheduled_penalty_terms)
        + LATE_COUNT_WEIGHT * sum(late_count_terms)
        + LATENESS_WEIGHT * sum(lateness_terms)
        + IDLE_WEIGHT * sum(finish_terms)
    )

    solver = cp_model.CpSolver()
    status = solver.solve(model)

    return (solver, status, presence, start, scheduled, combos_by_order,
            combos_by_machine, combos_by_worker, order_duration, deadlines)


def hour_offset_to_clock(hour_offset: float) -> str:
    total_minutes = round(hour_offset * 60)
    hh = BASE_HOUR + total_minutes // 60
    mm = total_minutes % 60
    return f"{hh:02d}:{mm:02d}"


# -----------------------------------------------------------------------
# CHANGE 3: unscheduled-order diagnostics.
#
# For every order that ended up unscheduled, work out WHY, using the
# combo data already computed during model-building:
#   - no capable machine at all             -> "No compatible machine"
#   - capable machine(s), but no worker's
#     shift overlaps enough                 -> "No worker available"
#   - a feasible combo existed in isolation,
#     but the solver didn't use it because
#     that machine/worker's time was taken
#     by other (usually higher-priority)
#     orders                                -> "Insufficient machine/worker
#                                               capacity"
# -----------------------------------------------------------------------
def explain_unscheduled(o, machines, combos_by_order):
    capable_machines = [m for m in machines if o["product"] in m["capabilities"]]

    if not capable_machines:
        return (f"No compatible machine available - no machine in the factory "
                f"has capability '{o['product']}'.")

    combos = combos_by_order[o["id"]]
    if not combos:
        return (f"No worker available - {len(capable_machines)} machine(s) "
                f"can make '{o['product']}', but no worker's shift overlaps "
                f"any of those machines' availability for the full "
                f"{o['duration_hours']}h required.")

    unique_machines = {k[1] for k in combos}
    unique_workers = {k[2] for k in combos}

    if len(unique_machines) <= 1:
        return (f"Insufficient machine capacity - only one compatible machine "
                f"exists for product '{o['product']}', and it was fully booked "
                f"by other (higher-priority or better-fitting) orders.")
    if len(unique_workers) <= 1:
        return (f"Insufficient worker capacity - only one worker could staff "
                f"this order, and they were fully booked by other orders.")
    return (f"Insufficient machine/worker capacity - a feasible slot existed "
            f"({len(unique_machines)} machine(s), {len(unique_workers)} worker(s)), "
            f"but all of them were claimed by other, higher-value orders.")


def print_schedule(data, solver, status, presence, start, scheduled,
                    combos_by_order, order_duration, deadlines):
    print("=" * 95)
    print("FACTORY SCHEDULE")
    print("=" * 95)
    print(f"Solver status: {solver.status_name(status)}\n")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # With optional orders this should essentially never happen, but
        # it's kept as a safety net (e.g. if the model itself is malformed).
        print("No solution could be found at all.")
        return [], []

    machine_name = {m["id"]: m["name"] for m in data["machines"]}
    worker_name = {w["id"]: w["name"] for w in data["workers"]}
    priorities = {o["id"]: o["priority"] for o in data["orders"]}

    print(f"{'Order':<8}{'Product':<9}{'Machine':<12}{'Worker':<12}{'Start':<8}{'End':<8}"
          f"{'Deadline':<10}{'Lateness':<10}{'Penalty':<9}{'Priority'}")
    print("-" * 95)

    scheduled_rows = []
    unscheduled_orders = []

    for o in data["orders"]:
        o_id = o["id"]
        sched_var = scheduled[o_id]
        is_scheduled = (not isinstance(sched_var, int)) and solver.value(sched_var) == 1

        if not is_scheduled:
            unscheduled_orders.append(o)
            continue

        chosen = None
        for key in combos_by_order[o_id]:
            if solver.value(presence[key]) == 1:
                chosen = key
                break

        _, m_id, w_id = chosen
        duration = order_duration[o_id]
        start_h = solver.value(start[chosen])
        end_h = start_h + duration
        deadline_h = deadlines[o_id]
        priority = priorities[o_id]

        lateness_h = max(0, end_h - deadline_h)
        penalty = priority * lateness_h

        scheduled_rows.append((o_id, o["product"], m_id, w_id, start_h, end_h,
                                deadline_h, lateness_h, penalty, priority))

        print(
            f"{o_id:<8}{o['product']:<9}{machine_name[m_id]:<12}{worker_name[w_id]:<12}"
            f"{hour_offset_to_clock(start_h):<8}{hour_offset_to_clock(end_h):<8}"
            f"{hour_offset_to_clock(deadline_h):<10}"
            f"{(str(lateness_h) + 'h' if lateness_h > 0 else '-'):<10}"
            f"{(str(penalty) if lateness_h > 0 else '-'):<9}"
            f"{priority}"
        )

    if not scheduled_rows:
        print("(no orders were scheduled)")

    return scheduled_rows, unscheduled_orders


def print_unscheduled(data, unscheduled_orders, combos_by_order):
    print()
    print("=" * 95)
    print("ORDERS THAT COULD NOT BE SCHEDULED")
    print("=" * 95)

    if not unscheduled_orders:
        print("(none - every order was scheduled)")
        return

    print(f"{'Order':<8}{'Product':<9}{'Priority':<10}{'Reason'}")
    print("-" * 95)
    for o in unscheduled_orders:
        reason = explain_unscheduled(o, data["machines"], combos_by_order)
        print(f"{o['id']:<8}{o['product']:<9}{o['priority']:<10}{reason}")


def print_statistics(data, scheduled_rows, unscheduled_orders):
    print()
    print("=" * 95)
    print("STATISTICS")
    print("=" * 95)

    total_orders = len(data["orders"])
    late_orders = [r for r in scheduled_rows if r[7] > 0]
    total_penalty = sum(r[8] for r in scheduled_rows)

    print(f"Total orders           : {total_orders}")
    print(f"Scheduled orders        : {len(scheduled_rows)}")
    print(f"Unscheduled orders      : {len(unscheduled_orders)}")
    print(f"Late (scheduled) orders : {len(late_orders)}")
    print(f"Total lateness penalty  : {total_penalty}")
    for r in late_orders:
        o_id, _, _, _, _, end_h, deadline_h, lateness_h, penalty, priority = r
        print(f"    - {o_id} (priority {priority}): finished {hour_offset_to_clock(end_h)}, "
              f"deadline {hour_offset_to_clock(deadline_h)}, {lateness_h}h late, penalty {penalty}")

    print("\nMachine utilization (scheduled orders only):")
    machine_hours = {}
    for r in scheduled_rows:
        m_id = r[2]
        duration = r[5] - r[4]
        machine_hours[m_id] = machine_hours.get(m_id, 0) + duration
    for m in data["machines"]:
        used = machine_hours.get(m["id"], 0)
        window = to_hour_offset(m["available"][1]) - to_hour_offset(m["available"][0])
        idle = window - used
        print(f"    {m['name']:<10}: {used}h busy / {window}h available -> {idle}h idle")


def main():
    data = load_data(DATA_PATH)
    (solver, status, presence, start, scheduled, combos_by_order,
     combos_by_machine, combos_by_worker, order_duration, deadlines) = build_and_solve(data)

    scheduled_rows, unscheduled_orders = print_schedule(
        data, solver, status, presence, start, scheduled, combos_by_order, order_duration, deadlines
    )
    print_unscheduled(data, unscheduled_orders, combos_by_order)
    print_statistics(data, scheduled_rows, unscheduled_orders)

    # --- NEW: export results for the LLM layer ---
    output_dict = build_output_dict(data, scheduled_rows, unscheduled_orders, combos_by_order)
    output_path = Path(__file__).parent / "scheduler_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2)
    print(f"\nScheduler output written to {output_path}")


def build_output_dict(data, scheduled_rows, unscheduled_orders, combos_by_order):
    """
    Convert the scheduler's results into a plain JSON-serializable dict.
    This is purely a reporting/export step - it reads values that
    build_and_solve() already produced; it does not re-solve or alter
    anything.
    """
    machine_name = {m["id"]: m["name"] for m in data["machines"]}
    worker_name = {w["id"]: w["name"] for w in data["workers"]}

    # --- Scheduled orders ---
    scheduled_out = []
    for (o_id, product, m_id, w_id, start_h, end_h,
         deadline_h, lateness_h, penalty, priority) in scheduled_rows:
        scheduled_out.append({
            "order_id": o_id,
            "product": product,
            "priority": priority,
            "machine_id": m_id,
            "machine_name": machine_name[m_id],
            "worker_id": w_id,
            "worker_name": worker_name[w_id],
            "start": hour_offset_to_clock(start_h),
            "end": hour_offset_to_clock(end_h),
            "deadline": hour_offset_to_clock(deadline_h),
            "lateness_hours": lateness_h,
            "penalty": penalty,
            "is_late": lateness_h > 0,
        })

    # --- Unscheduled orders + reasons (reuses existing diagnostics) ---
    unscheduled_out = []
    for o in unscheduled_orders:
        unscheduled_out.append({
            "order_id": o["id"],
            "product": o["product"],
            "priority": o["priority"],
            "reason": explain_unscheduled(o, data["machines"], combos_by_order),
        })

    # --- Machine utilization ---
    machine_hours = {}
    for row in scheduled_rows:
        m_id = row[2]
        duration = row[5] - row[4]
        machine_hours[m_id] = machine_hours.get(m_id, 0) + duration

    machine_utilization = []
    for m in data["machines"]:
        used = machine_hours.get(m["id"], 0)
        window = to_hour_offset(m["available"][1]) - to_hour_offset(m["available"][0])
        machine_utilization.append({
            "machine_id": m["id"],
            "machine_name": m["name"],
            "available_hours": window,
            "busy_hours": used,
            "idle_hours": window - used,
        })

    # --- Worker workload ---
    worker_hours = {}
    for row in scheduled_rows:
        w_id = row[3]
        duration = row[5] - row[4]
        worker_hours[w_id] = worker_hours.get(w_id, 0) + duration

    worker_workload = []
    for w in data["workers"]:
        worker_workload.append({
            "worker_id": w["id"],
            "worker_name": w["name"],
            "shift": w["shift"],
            "assigned_hours": worker_hours.get(w["id"], 0),
        })

    # --- Summary stats ---
    late_orders = [r for r in scheduled_rows if r[7] > 0]
    stats = {
        "total_orders": len(data["orders"]),
        "scheduled_count": len(scheduled_rows),
        "unscheduled_count": len(unscheduled_orders),
        "late_count": len(late_orders),
        "on_time_count": len(scheduled_rows) - len(late_orders),
        "total_lateness_penalty": sum(r[8] for r in scheduled_rows),
    }

    return {
        "scheduled_orders": scheduled_out,
        "unscheduled_orders": unscheduled_out,
        "machine_utilization": machine_utilization,
        "worker_workload": worker_workload,
        "statistics": stats,
    }


if __name__ == "__main__":
    main()