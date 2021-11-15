#pragma once
#include<Eigen/core>
#include<iostream>
#include"Callback.h"
#include"Config.h"
#include"NeuralNet.h"

class VerboseCallback : public Callback
{
	void post_training_batch(const NeuralNet* net, const Matrix& x, const Matrix& y)
	{
		const Scalar loss = net->get_output()->loss();
		std::cout << "[Epoch " << m_epoch_id << ", batch " << m_batch_id << "] loss = " << loss << std::endl;
		m_loss.push_back(loss);
	}

	void post_training_batch(const NeuralNet* net, const Matrix& x, const IntegerVector& y)
	{
		const Scalar loss = net->get_output()->loss();
		std::cout << "[Epoch " << m_epoch_id << ", batch " << m_batch_id << "] loss = " << loss << std::endl;
		m_loss.push_back(loss);
	}
};

