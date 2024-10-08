#pragma once
#define LEARNING_RATE 0.1

class CMLP
{
public:
	CMLP();
	~CMLP();

	// 입력노드의 수
	int m_iNuminNodes;
	int m_iNumOutNodes;
	int m_iNumHiddenLayer;   //hidden only
	int m_iNumTotalLayer;    // inputlayer + hiddenlayer + outputlayer
	int* m_NumNodes;         // [0] - input node, [1.] - hidden layer, [m.iNumHiddenLayer + 1], output layer

	double*** m_Weight;      // [시작layer][시작노드][연결노드]
	double** m_NodeOut;      // [layer][node]

	double* pinValue, * pOutValue;   // 입력레이어, 출력레이어
	double* pCorrectOutValue;        // 정답레이어
	
	double** m_ErrorGradient; // [layer][node]


	bool Create(int InNode, int* pHiddenNode, int OutNode, int NumHiddenLayer);
private:
	void initw();
	double ActivationFunc(double weightsum);
public:
	void Forward();
	void BackPopagationLearning();
};