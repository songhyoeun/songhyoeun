#include "MLP.h"
#include <malloc.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


CMLP::CMLP()
{
	int layer;

	m_iNuminNodes = 0;
	m_iNumOutNodes = 0;

	m_NumNodes = NULL;
	m_NodeOut = NULL;
	m_ErrorGradient = NULL;
	m_Weight = NULL;

	pinValue = NULL;
	pOutValue = NULL;
	pCorrectOutValue = NULL;
}

CMLP::~CMLP()
{
	int i, layer, snode, enode;

	if (m_NodeOut != NULL) {
		for (i = 0; i < (m_iNumTotalLayer + 1); i++)
			free(m_NodeOut[i]);
		free(m_NodeOut);
	}

	if (m_Weight != NULL) {
		for (layer = 0; layer < (m_iNumTotalLayer - 1); layer++) {
			if (m_Weight[layer] != NULL) {
				for (snode = 0; snode < m_NumNodes[layer] + 1; snode++)
					free(m_Weight[layer][snode]);
				free(m_Weight[layer]);
			}
		}
	}

	if (m_ErrorGradient != NULL) {
		for (layer = 0; layer < (m_iNumTotalLayer); layer++)
			free(m_ErrorGradient[layer]);
		free(m_ErrorGradient);
	}

	if (m_NumNodes != NULL)
		free(m_NumNodes);
}

bool CMLP::Create(int InNode, int* pHiddenNode, int OutNode, int NumHiddenLayer)
{
	int layer, snode, enode;

	m_iNuminNodes = InNode;
	m_iNumOutNodes = OutNode;
	m_iNumHiddenLayer = NumHiddenLayer;          // �Է�, ����� ����
	m_iNumTotalLayer = NumHiddenLayer + 2;       // ���� + �Է� + ���

	// m_iNuminNodes�� ���� �޸� �Ҵ�
	m_NumNodes = (int*)malloc((m_iNumTotalLayer + 1) * sizeof(int));   // ����(+1)

	m_NumNodes[0] = m_iNuminNodes;
	for (layer = 0; layer < m_iNumHiddenLayer; layer++)
		m_NumNodes[1 + layer] = pHiddenNode[layer];
	m_NumNodes[m_iNumTotalLayer - 1] = m_iNumOutNodes;   // ����� ����
	m_NumNodes[m_iNumTotalLayer] = m_iNumOutNodes;       // ���� ����

	// �� ��庰 ��� �޸� �Ҵ� = [layerno][nodeno]
	// �Է�: m_NodeOut[0][], ���: m_NodeOut[m_iNumTotalLayer - 1][]
	// ����: m_NodeOUt[m_iNumTotalLayer][]

	m_NodeOut = (double**)malloc((m_iNumTotalLayer + 1) * sizeof(double*));
	for (layer = 0; layer < m_iNumTotalLayer; layer++)
		m_NodeOut[layer] = (double*)malloc((m_NumNodes[layer] + 1) * sizeof(double));
	// ���� (��� ���� ���� ����, ���̿����� �ʿ������ ÷�ڴ� 1���� n����)
	m_NodeOut[m_iNumTotalLayer] = (double*)malloc((m_NumNodes[m_iNumTotalLayer - 1] + 1) * sizeof(double));

	// ����ġ �޸��Ҵ� m_Weight[���۷��̾�][���۳��][������]
	m_Weight = (double***)malloc((m_iNumTotalLayer - 1) * sizeof(double**));
	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++)

	return true;
}

void CMLP::initw()
{
	int layer, snode, enode;

	srand(time(NULL));
	for (layer = 0; layer < m_iNumTotalLayer; layer++) {
		for (snode = 0; snode <= m_NumNodes[layer]; snode++) {
			for (enode = 0; enode <= m_NumNodes[layer + 1]; enode++) {
				m_Weight[layer][snode][enode] = (double)rand() / RAND_MAX - 0.5;
			}
		}
	}
}

double CMLP::ActivationFunc(double weightsum)
{
	// if (weightsum > 0)
		// return 1.0;
	// else
		// return 0.0;
	return 1 / (1 + exp(-weightsum));
}


void CMLP::Forward()
{
	int layer, snode, enode;
	double wsum;

	for (layer = 0; layer < m_iNumTotalLayer - 1; layer++) {
		for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++) {
			wsum = 0.0;
			wsum += m_Weight[layer][0][enode] * 1;
			for (snode = 1; snode < m_NumNodes[layer]; snode++) {
				wsum += m_Weight[layer][snode][enode] * m_NodeOut[layer][snode];
			}

			m_NodeOut[layer + 1][enode] = ActivationFunc(wsum);
		}
	}
}


void CMLP::BackPopagationLearning()
{
	int layer;

	if (m_ErrorGradient == NULL) {
		m_ErrorGradient = (double**)malloc((m_iNumTotalLayer) * sizeof(double*));
		for (layer = 0; layer < m_iNumTotalLayer; layer++)
			m_ErrorGradient[layer] = (double*)malloc((m_NumNodes[layer] + 1) * sizeof(double));
	}

	int snode, enode, node;
	for (node = 1; node <= m_iNumOutNodes; node++) {
		m_ErrorGradient[m_iNumTotalLayer - 1][node] = (pCorrectOutValue[node] - m_NodeOut[m_iNumTotalLayer - 1][node])
			* m_NodeOut[m_iNumTotalLayer - 1][node] * (1 - m_NodeOut[layer][snode]);
	}

	for (layer = m_iNumTotalLayer - 2; layer >= 0; layer--) {
		for (snode = 1; snode <= m_NumNodes[layer]; snode++) {
			m_ErrorGradient[layer][snode] = 0.0;
			for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++) {
				m_ErrorGradient[layer][snode] += (m_ErrorGradient[layer + 1][enode] * m_Weight[layer][snode][enode]);
			}

			m_ErrorGradient[layer][snode] *= m_NodeOut[layer][snode] * (1 - m_NodeOut[layer][snode]);
		}
	}

	for (layer = m_iNumTotalLayer - 2; layer >= 0; layer--) {
		for (enode = 1; enode <= m_NumNodes[layer + 1]; enode++) {
			m_Weight[layer][0][enode] += (LEARNING_RATE * m_ErrorGradient[layer + 1][enode] * 1);
			for (snode = 1; snode <= m_NumNodes[layer]; snode++) {
				m_Weight[layer][snode][enode] += (LEARNING_RATE * m_ErrorGradient[layer + 1][enode] * m_NodeOut[layer][snode]);
			}
		}
	}
}
