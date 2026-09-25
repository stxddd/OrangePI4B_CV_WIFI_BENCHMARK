# Orange Pi 4B CV Wi-Fi Benchmark

Проект объединяет два сценария:

1. Orange Pi 4B 1 получает кадры с камеры или видеоисточника и передаёт их по Wi-Fi;
2. Orange Pi 4B 2 принимает кадры, запускает на них ONNX-моделеи компьютерного зрения и сравнивает производительность.

Цель проекта - измерить работу моделей в условиях, близких к реальному использованию: кадр должен пройти через Wi-Fi, быть восстановлен на Orange Pi и только после этого попасть в инференс.

## Состав проекта

В этом каталоге уже объединены:

- `sender.py` - передача кадров с камеры или видеофайла по UDP;
- `wifi_source.py` - сборка UDP-фрагментов, декодирование JPEG и измерение RTT;
- `benchmark_onnx_models.py` - общий бенчмарк ONNX-моделей для источников `video` и `wifi`;
- каталоги инференса и ONNX-модели из `CV_benchmark`.


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

| Устройство | Роль | Адрес по умолчанию |
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

### 1. Orange Pi 1

```bash
sudo apt update
sudo apt install -y network-manager openssh-server python3-opencv v4l-utils dnsmasq
sudo systemctl enable --now NetworkManager ssh
sudo nmcli radio wifi on
sudo nmcli device wifi hotspot ifname wlan0 con-name OrangePi-AP ssid OrangePi-Cam password 'OrangePiCam2026'
sudo nmcli connection modify OrangePi-AP ipv4.method shared ipv4.addresses 192.168.50.1/24 ipv6.method disabled connection.autoconnect yes
sudo nmcli connection up OrangePi-AP
```

### 2. Orange Pi 2

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

Подключение к Orange Pi A по SSH:

```bash
ssh orangepi@192.168.50.1
```

### 3. Запуск

На устройстве-источнике запускается передатчик с камерой:

```bash
python sender.py --source camera --receiver-ip 192.168.50.2
```

По умолчанию передатчик читает `/dev/video0`, передаёт кадры размером `320x240` с частотой `15 FPS` и качеством JPEG `75`. Устройство и параметры можно изменить:

```bash
python sender.py --source camera --camera /dev/video2 --fps 20 --jpeg-quality 80
```

В Windows номер камеры можно указать числом, например `--camera 0`.

Для передачи готового видеофайла используется тот же передатчик:

```bash
python sender.py source video video /path/to/video.mp4 
```

Файл передаётся от начала до конца, после чего передатчик завершается. В обоих режимах кадры кодируются в JPEG и отправляются по UDP.

## Запуск бенчмарка

Файловый источник:

```bash
python benchmark_onnx_models.py \
  --source video \
  --models yolo8n \
  --video /path/to/test.mp4 \
  --threads 4 \
  --warmup 20
```

Wi-Fi-источник на Orange Pi 4B:

```bash
python benchmark_onnx_models.py \
  --source wifi \
  --sender-ip 192.168.50.1 \
  --port 5000 \
  --control-port 5001 \
  --source-fps 15 \
  --models yolo8n \
  --max-frames 400 \
  --threads 4 \
  --warmup 20
```

Для запуска всех моделей из Wi-Fi-потока:

```bash
python benchmark_onnx_models.py \
  --source wifi \
  --sender-ip 192.168.50.1 \
  --threads 4 \
  --warmup 20 \
  --output-dir benchmark_results/all \
  --output-csv benchmark_results/all.csv
```

Доступные псевдонимы моделей:

```text
efficientdet nanodet picodet ssd_v2 leyolo_nano
yolo5n yolo8n yolo10n yolo11n yolo26n
```

По умолчанию обрабатываются первые `400` кадров. Значение `--max-frames 0` означает обработку до конца видео; для Wi-Fi нужно указать положительное количество кадров. При Wi-Fi-режиме кадры сначала накапливаются, затем используются всеми моделями.

## Метрики

Бенчмарк должен сохранять как минимум:

- число обработанных кадров;
- средний, минимальный и максимальный FPS;
- среднюю, минимальную и максимальную задержку инференса;
- время подготовки входа и декодирования результата, если эти этапы измеряются отдельно;
- Wi-Fi RTT;
- число потерянных или неполных кадров;
- параметры модели: размер файла, число параметров, оценку GMACs и входное разрешение.

Результаты сохраняются в CSV. При включённом выводе видео также сохраняется аннотированный ролик с рамками, confidence, FPS и latency.

Важно: latency инференса и Wi-Fi RTT - разные величины. RTT измеряет сетевой обмен контрольным пакетом, а latency модели измеряет обработку уже принятого кадра. Полная задержка кадра должна рассчитываться отдельно, если в протокол добавить временные метки отправки и приёма.

## Текущие ограничения

- UDP не гарантирует доставку и порядок пакетов; потеря фрагмента делает JPEG-кадр непригодным.
- Wi-Fi-режим накапливает кадры до запуска инференса, чтобы все модели получили одинаковую выборку.
- Текущий протокол не содержит timestamp кадра, поэтому RTT не является полной end-to-end задержкой.
- `wifi_rtt_ms` записывается в CSV отдельно от latency инференса.

Разные модели используют собственные preprocess/decode-функции, поэтому `wifi_source.py` отдаёт исходный BGR-кадр и не вмешивается в обработку модели.

## Исходные проекты

- `CV_benchmark` - источник ONNX-моделей, preprocess/decode и расчёт benchmark-метрик.
- `OrangePi4b_Connections/yolo_wifi_stream` - исходный UDP-проект, код которого перенесён в текущий каталог.