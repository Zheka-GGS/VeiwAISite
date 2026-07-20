# CIFAR-10 Image Classifier

Класифікатор зображень на основі CNN, навчений на CIFAR-10. Веб-інтерфейс для завантаження зображень.

## Як запустити

```bash
pip install -r requirements.txt
python app.py
```

Потім відкрийте http://localhost:5000

## Що робить

Завантажує зображення, перетворює на 32x32, подає в модель і повертає клас з ймовірністю.

## Моделі

- Custom CNN - своя архітектура
- VGG16 transfer learning - попередньо навчена VGG16 + власні шари

Найкраща модель зберігається в `cifar10_classifier.h5`

## Docker

```bash
docker-compose up --build
```
