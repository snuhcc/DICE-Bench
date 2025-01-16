import logging
from colorlog import ColoredFormatter

# 로거 및 핸들러 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 스트림 핸들러 생성
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# 색상 포매터 지정
formatter = ColoredFormatter(
    "%(log_color)s[%(levelname)s]%(reset)s %(blue)s%(name)s:%(reset)s %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'bold_red'
    },
)
handler.setFormatter(formatter)
logger.addHandler(handler)