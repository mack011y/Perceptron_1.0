package model

import (
	"math/rand"
)

// Perceptron представляет структуру однослойного перцептрона
type Perceptron struct {
	Weights []float64
	Bias    float64
}

// NewPerceptron создает новый перцептрон с заданным количеством признаков
func NewPerceptron(numFeatures int) *Perceptron {
	weights := make([]float64, numFeatures)

	// Инициализация весов небольшими случайными значениями
	for i := range weights {
		weights[i] = rand.NormFloat64() * 0.01
	}

	return &Perceptron{
		Weights: weights,
		Bias:    0.0,
	}
}

// NewPerceptronWithSeed создает новый перцептрон с заданным seed для инициализации весов
func NewPerceptronWithSeed(numFeatures int, seed int64) *Perceptron {
	weights := make([]float64, numFeatures)

	// Создаем локальный генератор случайных чисел с указанным seed
	source := rand.NewSource(seed)
	rng := rand.New(source)

	// Инициализация весов небольшими случайными значениями
	for i := range weights {
		weights[i] = rng.NormFloat64() * 0.01
	}

	return &Perceptron{
		Weights: weights,
		Bias:    0.0,
	}
}

// Predict выполняет прямой проход перцептрона для одного примера
func (p *Perceptron) Predict(x []float64) float64 {
	z := p.Bias
	for i, feature := range x {
		z += feature * p.Weights[i]
	}
	return Sigmoid(z) // Используем нашу точную реализацию сигмоиды
}

// PredictBatch выполняет предсказания для набора примеров
func (p *Perceptron) PredictBatch(X [][]float64) []float64 {
	predictions := make([]float64, len(X))
	for i, x := range X {
		predictions[i] = p.Predict(x)
	}
	return predictions
}
