class ViewingZoneProcessor:

    def __init__(
        self,
        x1=200,
        y1=150,
        x2=600,
        y2=500
    ):

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def process(self, tracks):

        people_inside = 0

        for track in tracks:

            if track["class"] != "person":
                continue

            x = track["center_x"]
            y = track["center_y"]

            if (
                self.x1 <= x <= self.x2
                and
                self.y1 <= y <= self.y2
            ):

                people_inside += 1

        return {
            "people_in_viewing_zone": people_inside
        }