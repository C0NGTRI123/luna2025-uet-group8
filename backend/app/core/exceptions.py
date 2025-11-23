from http import HTTPStatus


class CustomException(Exception):

    code = HTTPStatus.BAD_GATEWAY
    error_code = HTTPStatus.BAD_GATEWAY
    message = HTTPStatus.BAD_GATEWAY.description

    def __init__(self, message=None):
        if message:
            self.message = message
        super().__init__(self.message)


class InvalidFileFormatException(CustomException):
    code = HTTPStatus.BAD_REQUEST
    error_code = "INVALID_FILE_FORMAT"
    message = "File format is not valid .mha or .mhd format or missing series_uid"


class UnauthorizedException(CustomException):
    code = HTTPStatus.UNAUTHORIZED
    error_code = "UNAUTHORIZED"
    message = "Request missing or invalid Authorization: Bearer <token>"


class ForbiddenException(CustomException):
    code = HTTPStatus.FORBIDDEN
    error_code = "FORBIDDEN"
    message = "Account/Token is locked or exceeded daily API call limit"


class NotFoundException(CustomException):
    code = HTTPStatus.NOT_FOUND
    error_code = "NOT_FOUND"
    message = "Endpoint does not exist or Service Model is offline"


class ProcessingErrorException(CustomException):
    code = HTTPStatus.UNPROCESSABLE_ENTITY
    error_code = "PROCESSING_ERROR"
    message = "Internal model error (e.g., GPU memory overflow, image processing library error)"


class InternalServerErrorException(CustomException):
    code = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code = "INTERNAL_SERVER_ERROR"
    message = "Unidentified system server error"


class GatewayTimeoutException(CustomException):
    code = HTTPStatus.GATEWAY_TIMEOUT
    error_code = "GATEWAY_TIMEOUT"
    message = "Processing time exceeded 600 seconds"