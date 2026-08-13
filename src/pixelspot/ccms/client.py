class CCMSClient:

    def __init__(self, enabled=False):
        self.enabled = enabled

    def send(self, data):

        if not self.enabled:
            print("CCMS integration disabled.")
            return False

        print("CCMS data prepared for sending.")

        return True
    