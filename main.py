import numpy as np
import cv2
import time
import psutil
import threading

def simulate_fake_frame():
    """Simuluje náhodný RGB snímek z kamery (640x480)."""
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

def simulate_ai_processing(frame):
    """Simuluje zátěž jako by běžela AI."""
    small = cv2.resize(frame, (320, 240))
    blur = cv2.GaussianBlur(small, (15, 15), 0)
    edges = cv2.Canny(blur, 50, 150)
    _ = np.linalg.svd(np.random.rand(300, 300))  # Těžká matice
    return edges

def monitor_performance(stop_event):
    while not stop_event.is_set():
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory().percent
        print(f"[MONITOR] CPU: {cpu:.1f}% | RAM: {ram:.1f}%")

def main():
    print("[INFO] Spouštím simulovaný benchmark (bez kamery)")
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_performance, args=(stop_event,))
    monitor_thread.start()

    frame_count = 0
    start_time = time.time()
    DURATION_SEC = 10

    try:
        while True:
            now = time.time()
            if now - start_time > DURATION_SEC:
                break

            fake_frame = simulate_fake_frame()
            result = simulate_ai_processing(fake_frame)
            frame_count += 1

    finally:
        stop_event.set()
        monitor_thread.join()
        duration = time.time() - start_time
        print(f"[DONE] Zpracováno {frame_count} snímků za {duration:.2f} sekund (~{frame_count/duration:.1f} FPS)")

if __name__ == "__main__":
    main()
