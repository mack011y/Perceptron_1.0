package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"Perceptron/data"
	"Perceptron/model"
	"Perceptron/utils"
)

// Тестирование собственной реализации сигмоиды
func testSigmoid() {
	fmt.Println("\nСравнение реализаций сигмоидной функции:")
	fmt.Println("----------------------------------------------")
	fmt.Printf("%-10s %-15s %-15s %-15s\n", "z", "Стандартная", "Наша реализация", "Отклонение")

	// Стандартная сигмоида из библиотеки
	stdSigmoid := func(z float64) float64 {
		return 1.0 / (1.0 + math.Exp(-z))
	}

	// Тестирование на разных значениях z
	testValues := []float64{-10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10}
	for _, z := range testValues {
		standard := stdSigmoid(z)
		custom := model.Sigmoid(z)
		diff := math.Abs(standard - custom)

		fmt.Printf("%-10.2f %-15.10f %-15.10f %-15.10f\n", z, standard, custom, diff)
	}
	fmt.Println("----------------------------------------------")
}

func main() {
	// Устанавливаем случайный seed на основе текущего времени
	seed := time.Now().UnixNano()
	rand.Seed(seed)
	fmt.Printf("Инициализация с seed: %d\n", seed)

	// Тестируем собственную реализацию сигмоиды
	testSigmoid()

	fmt.Println("Обучение однослойного перцептрона")

	// Конфигурация генерации данных
	genConfig := &data.DataGenerationConfig{
		NumSamples:    10000,       // 10 тыс примеров для демонстрации
		NumFeatures:   30,          // 30 признаков
		WeightScale:   1.0,         // Стандартный масштаб весов
		FeatureScale:  1.0,         // Стандартный масштаб признаков
		PositiveRatio: 0.5,         // Сбалансированные классы
		RandomSeed:    int64(seed), // Используем тот же seed
	}

	// Генерация данных
	fmt.Println("Подготовка данных...")
	X, y := data.GenerateSyntheticData(genConfig)

	// Конфигурация разделения данных
	splitConfig := &data.SplitConfig{
		TrainRatio: 0.8,         // 80% на обучение, 20% на тест
		Shuffle:    true,        // Перемешиваем данные
		RandomSeed: int64(seed), // Используем тот же seed
	}

	// Разделение на обучающую и тестовую выборки
	trainX, trainY, testX, testY := data.SplitData(X, y, splitConfig)

	// Создание модели с использованием того же seed
	p := model.NewPerceptronWithSeed(genConfig.NumFeatures, seed)

	// Конфигурация обучения
	trainConfig := &model.TrainingConfig{
		LearningRate:   0.1,    // Коэффициент обучения
		Epochs:         100,    // Максимальное число эпох
		LogEvery:       10,     // Вывод каждые 10 эпох
		EarlyStopDelta: 0.0001, // Остановка при малом изменении потери
	}

	// Обучение модели
	fmt.Println("Создание и обучение модели...")
	start := time.Now()
	p.Train(trainX, trainY, trainConfig)
	duration := time.Since(start)

	// Конфигурация метрик с расширенной статистикой
	metricsConfig := &utils.MetricsConfig{
		Threshold:       0.5,  // Стандартный порог классификации
		ExtendedMetrics: true, // Вычислять дополнительные метрики
	}

	// Оценка модели
	result := utils.EvaluateModel(p, testX, testY, metricsConfig)

	// Вывод результатов
	fmt.Printf("Обучение завершено за %v\n", duration)
	fmt.Printf("Точность на тестовых данных: %.2f%%\n", result.Accuracy)

	if metricsConfig.ExtendedMetrics {
		fmt.Printf("Precision: %.4f, Recall: %.4f, F1: %.4f\n",
			result.Precision, result.Recall, result.F1)
	}

	// Выводим матрицу ошибок
	cm := result.CM
	fmt.Println("\nМатрица ошибок:")
	fmt.Printf("TN: %d | FP: %d\n", cm[0][0], cm[0][1])
	fmt.Printf("FN: %d | TP: %d\n", cm[1][0], cm[1][1])
}
