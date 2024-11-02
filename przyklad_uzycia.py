import main_model
import threading


def run_recognition():
    main_model.run_recognition()

# uruchamianie rozpoznawanie gestów w nowej pętli, bo działa w nieskończonej pętli
realtime_thread = threading.Thread(target=run_recognition)
realtime_thread.daemon = True
realtime_thread.start()

running = True
while running:      #pętla głowna programu/aplikacji/GUI
    from main_model import gesture
    print(gesture)


