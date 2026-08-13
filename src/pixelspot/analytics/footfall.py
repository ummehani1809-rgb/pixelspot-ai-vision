import time


class FootfallProcessor:

    def __init__(self, line_y=400):

        # Position of counting line
        self.line_y = line_y

        # Total crossing events
        self.people_entered = 0
        self.people_exited = 0

        # Highest occupancy observed
        self.peak_count = 0

        # History for active tracking IDs
        self.track_history = {}

        # How long to remember a temporarily
        # missing tracking ID
        self.track_timeout = 5.0

    # ==================================================
    # DETERMINE WHICH SIDE OF THE LINE
    # ==================================================

    def get_side(self, center_y):

        if center_y < self.line_y:
            return "above"

        return "below"

    # ==================================================
    # PROCESS CURRENT TRACKS
    # ==================================================

    def process(self, tracks):

        current_time = time.time()

        current_ids = set()

        events = []

        # ==================================================
        # PROCESS EVERY TRACK
        # ==================================================

        for track in tracks:

            track_id = track["id"]

            center_y = track["center_y"]

            current_ids.add(track_id)

            current_side = self.get_side(center_y)

            # ==================================================
            # NEW TRACK
            # ==================================================

            if track_id not in self.track_history:

                self.track_history[track_id] = {

                    "previous_y": center_y,

                    "side": current_side,

                    "last_seen": current_time
                }

                continue

            # ==================================================
            # EXISTING TRACK
            # ==================================================

            previous_side = (
                self.track_history[track_id]["side"]
            )

            # ==================================================
            # ENTER
            # ==================================================

            if (
                previous_side == "above"
                and current_side == "below"
            ):

                self.people_entered += 1

                events.append({
                    "type": "ENTER",
                    "track_id": track_id
                })

            # ==================================================
            # EXIT
            # ==================================================

            elif (
                previous_side == "below"
                and current_side == "above"
            ):

                self.people_exited += 1

                events.append({
                    "type": "EXIT",
                    "track_id": track_id
                })

            # ==================================================
            # UPDATE HISTORY
            # ==================================================

            self.track_history[track_id] = {

                "previous_y": center_y,

                "side": current_side,

                "last_seen": current_time
            }

        # ==================================================
        # REMOVE EXPIRED TRACKS
        # ==================================================

        expired_ids = []

        for track_id, data in self.track_history.items():

            if track_id not in current_ids:

                time_missing = (
                    current_time - data["last_seen"]
                )

                if time_missing > self.track_timeout:

                    expired_ids.append(track_id)

        for track_id in expired_ids:

            del self.track_history[track_id]

        # ==================================================
        # CURRENT OCCUPANCY
        # ==================================================

        current_occupancy = max(
            0,
            self.people_entered - self.people_exited
        )

        # ==================================================
        # PEAK
        # ==================================================

        self.peak_count = max(
            self.peak_count,
            current_occupancy
        )

        # ==================================================
        # RETURN RESULTS
        # ==================================================

        return {

            "people_entered": self.people_entered,

            "people_exited": self.people_exited,

            "current_occupancy": current_occupancy,

            "peak_count": self.peak_count,

            "events": events
        }