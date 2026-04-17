import easyocr

# Strategy Interface
class OCRStrategy:
    def detect_text(self, image):
        raise NotImplementedError


# Concrete Strategy
class EasyOCRStrategy(OCRStrategy):
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)

    def detect_text(self, image):
        return self.reader.readtext(image)


# Context
class LicensePlateDetector:
    def __init__(self, strategy):
        self.strategy = strategy

    def detect(self, image):
        return self.strategy.detect_text(image)