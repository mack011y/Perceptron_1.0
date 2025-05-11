package data

import (
	"math/rand"
)

// SplitConfig содержит параметры разделения данных на обучающую и тестовую выборки
type SplitConfig struct {
	// Доля данных для обучающей выборки
	TrainRatio float64

	// Перемешивать ли данные перед разделением
	Shuffle bool

	// Использовать ли стратификацию (сохранение распределения классов)
	Stratify bool

	// Seed для генератора случайных чисел
	RandomSeed int64
}

// DefaultSplitConfig создает конфигурацию разделения с настройками по умолчанию
func DefaultSplitConfig() *SplitConfig {
	return &SplitConfig{
		TrainRatio: 0.8,
		Shuffle:    true,
		Stratify:   false, // Стратификация отключена по умолчанию
		RandomSeed: 42,    // Фиксированный seed
	}
}

// SplitData разделяет данные на обучающую и тестовую выборки
func SplitData(X [][]float64, y []float64, config *SplitConfig) ([][]float64, []float64, [][]float64, []float64) {
	// Используем значения по умолчанию, если конфигурация не указана
	if config == nil {
		config = DefaultSplitConfig()
	}

	numSamples := len(X)
	numTrain := int(float64(numSamples) * config.TrainRatio)

	// Создаем локальный генератор случайных чисел с указанным seed
	source := rand.NewSource(config.RandomSeed)
	rng := rand.New(source)

	// Получаем индексы для разделения
	var indices []int
	if config.Shuffle {
		indices = rng.Perm(numSamples)
	} else {
		indices = make([]int, numSamples)
		for i := range indices {
			indices[i] = i
		}
	}

	trainX := make([][]float64, numTrain)
	trainY := make([]float64, numTrain)
	testX := make([][]float64, numSamples-numTrain)
	testY := make([]float64, numSamples-numTrain)

	// Заполняем обучающую выборку
	for i := 0; i < numTrain; i++ {
		idx := indices[i]
		trainX[i] = X[idx]
		trainY[i] = y[idx]
	}

	// Заполняем тестовую выборку
	for i := 0; i < numSamples-numTrain; i++ {
		idx := indices[numTrain+i]
		testX[i] = X[idx]
		testY[i] = y[idx]
	}

	return trainX, trainY, testX, testY
}
