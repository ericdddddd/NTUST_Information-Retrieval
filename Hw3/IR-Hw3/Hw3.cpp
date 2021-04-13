#include <iostream>
#include <string>
#include<sstream>
#include <vector>

using namespace std;

void split_doc(string &doc ,vector<string>& vec) /* ����string �� vector */
{
	stringstream input(doc);
	string result;
	while (getline(input, result, ' '))
		vec.push_back(result);
	/*
	for (int i = 0; i < vec.size(); i++) 
		cout << vec[i] << endl; */
}
int data_load(vector<string> &doc , vector<string> &ret_doc) //�x�s�峹�æ^�Ǧ@�n���X��
{
	int numOfquery;
	cin >> numOfquery;
	cin.get(); // �N�o��Ʀr�᪺ '\n' Ū�� �A�_�h�|�v�T���򪺿�J
	for (int loop = 0; loop < numOfquery; loop++)
	{
		string document;
		string retrieve_doc;
		getline(cin, document); // ���Ū������\n
		doc.push_back(document);
		getline(cin, retrieve_doc);
		ret_doc.push_back(retrieve_doc);
	}
	return numOfquery;
}
bool inRelevantDoc(int &num ,string& doc_id, vector<string>& rel_doc) /*judge ������ find() */
{
	for (unsigned int index = 0; index < rel_doc.size(); index++)
	{
		if (doc_id == rel_doc[index])
		{
			num = index;
			return true;
		}
	}
	return false;
}
double cal_MAP(vector<string> &doc, vector<string> &rel_doc)
{
	double length = 1; // �p�⨫��ĴX�g
	double doc_count = 0; // �p��w�����峹�X�{�b������󪺼ƶq
	int query_size = rel_doc.size();/* ***** */
	vector <double> val;
	for (unsigned int index = 0; index < doc.size(); index++)
	{
		int num = 0;
		if (inRelevantDoc(num, doc[index], rel_doc))
		{
			doc_count++;
			rel_doc.erase(rel_doc.begin() + num);
			val.push_back(doc_count / length);
		}
		length++;
	}
	double sum = 0;
	for (unsigned int i = 0; i < val.size(); i++)
		sum += val[i];
	sum /= query_size; /*�̫ᵲ�G�O�����峹���g�ơA���O�˯��᪺�峹���X�g*/
	return sum;
}
int main()
{
	vector<string> docList;
	vector<string> ret_docList;
	vector<double> ans;
	int size = data_load(docList, ret_docList);

	for (int index = 0; index < size; index++)
	{
		vector<string> doc;
		vector<string> ret_doc;
		split_doc(docList[index], doc);
		split_doc(ret_docList[index], ret_doc);
		double doc_MAP = cal_MAP(doc, ret_doc);
		ans.push_back(doc_MAP);
	}
	double total = 0;
	for (unsigned int i = 0; i < ans.size(); i++)
		total += ans[i];
	total /= size;
	double rounding = (int)(10000.0 * total + 0.5) / 10000.0;
	cout << rounding;
}