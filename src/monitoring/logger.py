"""
Production Structured Logger
Emits JSON/Structured log entries without persisting sensitive video frames.
"""
import time
import json
import logging

class StructuredISLLogger:
    def __init__(self, log_file: str = "reports/production_inference.log"):
        self.logger = logging.getLogger("ISLProduction")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def log_event(self, session_id: str, detector_status: str, classifier_status: str,
                  prediction: str, confidence: float, latency_ms: float, error_type: str = "None"):
        entry = {
            "timestamp": time.time(),
            "session_id": session_id,
            "detector_status": detector_status,
            "classifier_status": classifier_status,
            "prediction": prediction,
            "confidence": round(float(confidence), 4),
            "latency_ms": round(float(latency_ms), 2),
            "error_type": error_type
        }
        self.logger.info(json.dumps(entry))
