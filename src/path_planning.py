from __future__ import annotations

import math
from typing import List

from src.models import CarPose, Cone, Path2D


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx * dx + dy * dy)


class PathPlanning:
    """Student-implemented path planner.

    You are given the car pose and an array of detected cones, each cone with (x, y, color)
    where color is 0 for yellow (right side) and 1 for blue (left side). The goal is to
    generate a sequence of path points that the car should follow.

    Implement ONLY the generatePath function.
    """

    def __init__(self, car_pose: CarPose, cones: List[Cone]):
        self.car_pose = car_pose
        self.cones = cones

    def generatePath(self) -> Path2D:
        """Return a list of path points (x, y) in world frame.

        Requirements and notes:
        - Cones: color==0 (yellow) are on the RIGHT of the track; color==1 (blue) are on the LEFT.
        - You may be given 2, 1, or 0 cones on each side.
        - Use the car pose (x, y, yaw) to seed your path direction if needed.
        - Return a drivable path that stays between left (blue) and right (yellow) cones.
        - The returned path will be visualized by PathTester.

        The path can contain as many points as you like, but it should be between 5-10 meters,
        with a step size <= 0.5. Units are meters.

        Replace the placeholder implementation below with your algorithm.
        """
        cx = self.car_pose.x
        cy = self.car_pose.y
        yaw = self.car_pose.yaw

        # store cones in arrays (1 = blue/left, 0 = yellow/right)
        blue: List[Cone] = []
        yellow: List[Cone] = []
        for cone in self.cones:
            if cone.color == 1:
                blue.append(cone)
            else:
                yellow.append(cone)

        # Sort cones by distance from the car
        def dist_cone_to_car(cone: Cone) -> float:
            return distance(cx, cy, cone.x, cone.y)

        blue.sort(key=dist_cone_to_car)
        yellow.sort(key=dist_cone_to_car)

        d_half = 1.0  # Assumed half track width 
        targets: List[tuple[float, float]] = []

        # Case #1 \ Cones on both sides (pair them and get midpoints)
        if len(blue) > 0 and len(yellow) > 0:
            if len(blue) <= len(yellow):
                short_list, long_list = blue, yellow
                is_short_blue = True
            else:
                short_list, long_list = yellow, blue
                is_short_blue = False

            used_indices: List[int] = []
            pairs = []
            for sc in short_list:
                best_idx = -1
                best_dist = 9999999
                for i in range(len(long_list)):
                    if i not in used_indices:
                        d = distance(sc.x, sc.y, long_list[i].x, long_list[i].y)
                        if d < best_dist:
                            best_dist = d
                            best_idx = i
                used_indices.append(best_idx)
                pairs.append((sc, long_list[best_idx]))

            # Midpoint of first pair
            sc0, lc0 = pairs[0]
            if is_short_blue:
                b0, y0 = sc0, lc0
            else:
                b0, y0 = lc0, sc0
            targets.append(((b0.x + y0.x) / 2.0, (b0.y + y0.y) / 2.0))

            # Project remaining unpaired cone 
            v_yb = (b0.x - y0.x, b0.y - y0.y)
            for i in range(len(long_list)):
                if i not in used_indices:
                    lc = long_list[i]
                    if is_short_blue:
                        targets.append((lc.x + 0.5 * v_yb[0], lc.y + 0.5 * v_yb[1]))
                    else:
                        targets.append((lc.x - 0.5 * v_yb[0], lc.y - 0.5 * v_yb[1]))

            # Midpoints of any additional pairs
            for pair_idx in range(1, len(pairs)):
                sc_k, lc_k = pairs[pair_idx]
                targets.append(((sc_k.x + lc_k.x) / 2.0, (sc_k.y + lc_k.y) / 2.0))

        # Case 2: Only blue cones   
        elif len(blue) > 0:
            if len(blue) >= 2:
                for i in range(len(blue) - 1):
                    tx = blue[i + 1].x - blue[i].x
                    ty = blue[i + 1].y - blue[i].y
                    l = distance(blue[i].x, blue[i].y, blue[i + 1].x, blue[i + 1].y)
                    if l > 0.00001:
                        nx = ty / l
                        ny = -tx / l
                        if i == 0:
                            targets.append((blue[0].x + d_half * nx, blue[0].y + d_half * ny))
                        targets.append((blue[i + 1].x + d_half * nx, blue[i + 1].y + d_half * ny))
            if len(targets) == 0:
                ux = blue[0].x - cx
                uy = blue[0].y - cy
                l = distance(cx, cy, blue[0].x, blue[0].y)
                if l > 0.00001:
                    nx = uy / l
                    ny = -ux / l
                    targets.append((blue[0].x + d_half * nx, blue[0].y + d_half * ny))
                else:
                    targets.append((blue[0].x + d_half * math.sin(yaw), blue[0].y - d_half * math.cos(yaw)))

        # Case 3: Only yellow cones 
        elif len(yellow) > 0:
            if len(yellow) >= 2:
                for i in range(len(yellow) - 1):
                    tx = yellow[i + 1].x - yellow[i].x
                    ty = yellow[i + 1].y - yellow[i].y
                    l = distance(yellow[i].x, yellow[i].y, yellow[i + 1].x, yellow[i + 1].y)
                    if l > 0.00001:
                        nx = -ty / l
                        ny = tx / l
                        if i == 0:
                            targets.append((yellow[0].x + d_half * nx, yellow[0].y + d_half * ny))
                        targets.append((yellow[i + 1].x + d_half * nx, yellow[i + 1].y + d_half * ny))
            if len(targets) == 0:
                ux = yellow[0].x - cx
                uy = yellow[0].y - cy
                l = distance(cx, cy, yellow[0].x, yellow[0].y)
                if l > 0.00001:
                    nx = -uy / l
                    ny = ux / l
                    targets.append((yellow[0].x + d_half * nx, yellow[0].y + d_half * ny))
                else:
                    targets.append((yellow[0].x - d_half * math.sin(yaw), yellow[0].y + d_half * math.cos(yaw)))

        # Sort targets along driving direction from car
        def dist_target_to_car(p: tuple[float, float]) -> float:
            return distance(cx, cy, p[0], p[1])

        targets.sort(key=dist_target_to_car)
        waypoints = [(cx, cy)] + targets

        # Extend waypoints to reach target length (7m)
        target_len = 7.0
        cumulative_len = 0.0
        for i in range(len(waypoints) - 1):
            cumulative_len += distance(waypoints[i][0], waypoints[i][1], waypoints[i + 1][0], waypoints[i + 1][1])

        if len(waypoints) == 1:
            # Case #4 / No cones (drive along yaw)
            waypoints.append((cx + target_len * math.cos(yaw), cy + target_len * math.sin(yaw)))
        elif cumulative_len < target_len:
            dx = waypoints[-1][0] - waypoints[-2][0]
            dy = waypoints[-1][1] - waypoints[-2][1]
            l = distance(waypoints[-2][0], waypoints[-2][1], waypoints[-1][0], waypoints[-1][1])
            rem = target_len - cumulative_len
            if l > 0.00001:
                waypoints.append((waypoints[-1][0] + (dx / l) * rem, waypoints[-1][1] + (dy / l) * rem))
            else:
                waypoints.append((waypoints[-1][0] + rem * math.cos(yaw), waypoints[-1][1] + rem * math.sin(yaw)))

        # Resample waypoints  
        step = 0.25
        dists = [0.0]
        for i in range(len(waypoints) - 1):
            d = distance(waypoints[i][0], waypoints[i][1], waypoints[i + 1][0], waypoints[i + 1][1])
            dists.append(dists[-1] + d)

        total_dist = dists[-1]
        path: Path2D = [(cx, cy)]
        s = step
        seg_idx = 0
        while s <= total_dist:
            while seg_idx < len(dists) - 1 and dists[seg_idx + 1] < s:
                seg_idx += 1
            seg_start = dists[seg_idx]
            seg_len = dists[seg_idx + 1] - dists[seg_idx]
            t = (s - seg_start) / seg_len if seg_len > 0.00001 else 0.0
            x = waypoints[seg_idx][0] + t * (waypoints[seg_idx + 1][0] - waypoints[seg_idx][0])
            y = waypoints[seg_idx][1] + t * (waypoints[seg_idx + 1][1] - waypoints[seg_idx][1])
            path.append((x, y))
            s += step

        return path
