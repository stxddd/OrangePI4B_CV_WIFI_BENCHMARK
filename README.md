# Orange Pi 4B CV Wi-Fi Benchmark

Проект объединяет два сценария:

1. Orange Pi 4B 1 получает кадры с камеры или видеоисточника и передаёт их по Wi-Fi;
2. Orange Pi 4B 2 принимает кадры, запускает на них ONNX-моделеи компьютерного зрения и сравнивает производительность.

Цель проекта - измерить работу моделей в условиях, близких к реальному использованию: кадр должен пройти через Wi-Fi, быть восстановлен на Orange Pi и только после этого попасть в инференс.



## Архитектура

```text
Компьютер-источник                         Orange Pi 4B
------------------                         ------------
Камера / видео
      |
      v
OpenCV -> JPEG -> UDP-фрагменты ----------> UDP receiver
                                                 |
                                                 v
                                      сборка JPEG -> BGR frame
                                                 |
                                                 v
                              preprocess -> ONNX Runtime -> decode
                                                 |
                                                 v
                                      метрики и результаты
```

Передача использует два UDP-порта:

- `5000` - фрагменты видеокадров;
- `5001` - контрольные пакеты для измерения Wi-Fi RTT.


| Поле | Размер | Назначение |
|---|---:|---|
| `magic` | 4 байта | `OPV1` |
| `frame_id` | 4 байта | номер кадра |
| `chunk_id` | 2 байта | номер фрагмента |
| `chunk_count` | 2 байта | количество фрагментов кадра |

JPEG-фрагменты обычно имеют размер `1200` байт.

## Предполагаемое распределение устройств

| Устройство | Роль | Адрес |
|---|---|---|
| Orange Pi 4B 1 | источник кадров и UDP-передатчик | `192.168.50.1` |
| Orange Pi 4B 2 | приём кадров и инференс | `192.168.50.2` |


## Подготовка Orange Pi 4B

```bash
sudo apt update
sudo apt install -y network-manager openssh-server python3-venv python3-opencv libopenblas-dev

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


## Настройка Wi-Fi

Пример схемы с точкой доступа на устройстве-источнике:

### 1. Orange Pi 4B 1

Установка зависимостей 
```bash
sudo apt update
sudo apt install -y network-manager openssh-server python3-opencv v4l-utils dnsmasq
sudo systemctl enable --now NetworkManager ssh
sudo nmcli radio wifi on
sudo nmcli device wifi hotspot ifname wlan0 con-name OrangePi-AP ssid OrangePi-Cam password 'OrangePiCam2026'
sudo nmcli connection modify OrangePi-AP ipv4.method shared ipv4.addresses 192.168.50.1/24 ipv6.method disabled connection.autoconnect yes
sudo nmcli connection up OrangePi-AP
```

### 2. Orange Pi 4B 2

Установка зависимостей 
```bash
sudo apt update
sudo apt install -y network-manager openssh-server python3-venv python3-opencv libopenblas-dev
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Подключение к точке доступа
```bash
sudo systemctl enable --now NetworkManager ssh
sudo nmcli radio wifi on
sudo nmcli device wifi connect OrangePi-Cam password 'OrangePiCam2026' ifname wlan0 name OrangePi-Client
sudo nmcli connection modify OrangePi-Client ipv4.method manual ipv4.addresses 192.168.50.2/24 ipv4.gateway 192.168.50.1 ipv4.dns 192.168.50.1 ipv6.method disabled connection.autoconnect yes
sudo nmcli connection up OrangePi-Client
```

На этом шаге ОБА устройства будут находится в одной изолированной сети.


### 3. Запуск

Сначала на приёмнике запускается benchmark

```bash
python benchmark_onnx_models.py
```

Затем на источнике запускается передатчик:

```bash
python sender.py camera
```

Для передачи видеофайла:

```bash
python sender.py video /path/to/video.mp4
```

Чтобы изменить количество кадров:

```bash
python benchmark_onnx_models.py --max-frames 50
```

Передатчик использует фиксированные настройки: камера `/dev/video0`, размер `320x240`, частота `15 FPS`, JPEG quality `75`, приёмник `192.168.50.2`.

## Метрики

Бенчмарк должен сохранять как минимум:

- число обработанных кадров;
- средний, минимальный и максимальный FPS;
- среднюю, минимальную и максимальную задержку инференса;
- время подготовки входа и декодирования результата, если эти этапы измеряются отдельно;
- Wi-Fi RTT;
- число потерянных или неполных кадров;
- параметры модели: размер файла, число параметров, оценку GMACs и входное разрешение.

Результаты сохраняются в `benchmark_results/results.csv` и в аннотированные ролики по каждой модели.



