package data

import (
	"math"
	"math/rand"
	"time"
)

func init() {
	// Инициализация генератора случайных чисел в init
	// Будет переопределена, если предоставлен RandomSeed
	rand.Seed(time.Now().UnixNano())
}

// DataGenerationConfig содержит параметры для генерации синтетических данных
type DataGenerationConfig struct {
	// Количество примеров для генерации
	NumSamples int

	// Количество признаков для каждого примера
	NumFeatures int

	// Масштаб для весов (стандартное отклонение)
	WeightScale float64

	// Масштаб для признаков (стандартное отклонение)
	FeatureScale float64

	// Соотношение классов (доля положительных примеров)
	PositiveRatio float64

	// Seed для генератора случайных чисел
	RandomSeed int64
}

// DefaultDataGenerationConfig создает конфигурацию с настройками по умолчанию
func DefaultDataGenerationConfig() *DataGenerationConfig {
	return &DataGenerationConfig{
		NumSamples:    1000000,
		NumFeatures:   30,
		WeightScale:   1.0,
		FeatureScale:  1.0,
		PositiveRatio: 0.5, // Сбалансированные классы
		RandomSeed:    time.Now().UnixNano(),
	}
}

// GenerateSyntheticData создает синтетические данные для задачи бинарной классификации
func GenerateSyntheticData(config *DataGenerationConfig) ([][]float64, []float64) {
	// Используем значения по умолчанию, если конфигурация не указана
	if config == nil {
		config = DefaultDataGenerationConfig()
	}

	// Создаем локальный генератор случайных чисел с указанным seed
	source := rand.NewSource(config.RandomSeed)
	rng := rand.New(source)

	X := make([][]float64, config.NumSamples)
	y := make([]float64, config.NumSamples)

	// Генерируем случайные веса для имитации реальных зависимостей
	trueWeights := make([]float64, config.NumFeatures)
	for i := range trueWeights {
		trueWeights[i] = rng.NormFloat64() * config.WeightScale
	}
	trueBias := rng.NormFloat64() * config.WeightScale

	// Генерируем данные
	for i := 0; i < config.NumSamples; i++ {
		X[i] = make([]float64, config.NumFeatures)

		// Генерируем признаки
		for j := 0; j < config.NumFeatures; j++ {
			X[i][j] = rng.NormFloat64() * config.FeatureScale
		}

		// Вычисляем вероятность дефолта на основе линейной комбинации
		z := trueBias
		for j := 0; j < config.NumFeatures; j++ {
			z += X[i][j] * trueWeights[j]
		}

		// Применяем сигмоиду для получения вероятности
		probability := 1.0 / (1.0 + math.Exp(-z))

		// Определяем класс на основе вероятности
		if rng.Float64() < probability {
			y[i] = 1.0
		} else {
			y[i] = 0.0
		}
	}

	return X, y
}
