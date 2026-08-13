class VehicleProcessor:

    VEHICLE_CLASSES = {
        "car",
        "motorcycle",
        "bus",
        "truck"
    }

    def process(self, tracks):

        counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0
        }

        for track in tracks:

            class_name = track["class"]

            if class_name in counts:

                counts[class_name] += 1

        counts["total_vehicles"] = sum(
            counts.values()
        )

        return counts
    