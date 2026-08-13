class Aggregator:

    def aggregate(
        self,
        footfall_result,
        vehicle_result,
        viewing_result
    ):

        return {

            "footfall": footfall_result,

            "vehicles": vehicle_result,

            "viewing_zone": viewing_result
        }