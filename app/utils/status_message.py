_STATUS_MESSAGES = {
    201: "success",
    400: "invalid_request",
    403: "unauthorized_server",
    428: "embedding_required",
    500: "internal_server_error"
}

def get_message_by_status(status_code: int) -> str:
    return _STATUS_MESSAGES.get(status_code, "unknown_status_message")