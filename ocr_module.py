from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")


def extract_image_text(image_path):

    result = ocr.ocr(image_path, cls=True)
    result = ' '.join([i[-1][0] for i in result[0]])
    return result