# config/settings.py

# 네트워크 설정
SERVER_IP = "127.0.0.1"  # 로컬호스트 (자기 자신)
SERVER_PORT = 5050       # 일반적으로 비어있는 포트 번호

# AI 알고리즘 설정 (초안용 임계값)
CONF_THRESHOLD_HIGH = 0.7  # 고화질 전송 기준
CONF_THRESHOLD_LOW = 0.3   # 저화질 전송/전송 제외 기준

# 데이터 설정
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720