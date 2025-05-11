package model

import (
	"fmt"
)

// TrainingConfig содержит параметры обучения модели
type TrainingConfig struct {
	// Коэффициент обучения
	LearningRate float64

	// Максимальное количество эпох обучения
	Epochs int

	// Периодичность вывода логов (каждые N эпох)
	LogEvery int

	// Критерий ранней остановки (минимальное изменение потери)
	EarlyStopDelta float64
}

// DefaultTrainingConfig создает конфигурацию обучения с настройками по умолчанию
func DefaultTrainingConfig() *TrainingConfig {
	return &TrainingConfig{
		LearningRate:   0.1,
		Epochs:         100,
		LogEvery:       10,
		EarlyStopDelta: 0.0001,
	}
}

// Train обучает перцептрон на данных с использованием конфигурации
func (p *Perceptron) Train(X [][]float64, y []float64, config *TrainingConfig) {
	numSamples := len(X)

	// Используем значения по умолчанию, если конфигурация не указана
	if config == nil {
		config = DefaultTrainingConfig()
	}

	prevLoss := float64(0)

	for epoch := 0; epoch < config.Epochs; epoch++ {
		// Прямой проход
		predictions := p.PredictBatch(X)

		// Вычисление функции потерь
		loss := CrossEntropy(y, predictions)

		// Проверка на раннюю остановку
		if epoch > 0 && (prevLoss-loss) < config.EarlyStopDelta {
			fmt.Printf("Досрочная остановка на эпохе %d: изменение потери %.6f < %.6f\n",
				epoch, prevLoss-loss, config.EarlyStopDelta)
			break
		}
		prevLoss = loss

		// Обновление весов
		for j := range p.Weights {
			gradient := 0.0
			for i := 0; i < numSamples; i++ {
				gradient += (predictions[i] - y[i]) * X[i][j]
			}
			gradient /= float64(numSamples)
			p.Weights[j] -= config.LearningRate * gradient
		}

		// Обновление смещения
		biasGradient := 0.0
		for i := 0; i < numSamples; i++ {
			biasGradient += (predictions[i] - y[i])
		}
		biasGradient /= float64(numSamples)
		p.Bias -= config.LearningRate * biasGradient

		// Вывод информации о процессе обучения
		if epoch%config.LogEvery == 0 {
			fmt.Printf("Эпоха %d: потеря = %.6f\n", epoch, loss)
		}
	}
}
