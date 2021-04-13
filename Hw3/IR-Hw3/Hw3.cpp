#include <iostream>
#include <string>
#include<sstream>
#include <vector>

using namespace std;

void split_doc(string &doc ,vector<string>& vec) /* 切割string 至 vector */
{
	stringstream input(doc);
	string result;
	while (getline(input, result, ' '))
		vec.push_back(result);
	/*
	for (int i = 0; i < vec.size(); i++) 
		cout << vec[i] << endl; */
}
int data_load(vector<string> &doc , vector<string> &ret_doc) //儲存文章並回傳共要做幾次
{
	int numOfquery;
	cin >> numOfquery;
	cin.get(); // 將得到數字後的 '\n' 讀取 ，否則會影響後續的輸入
	for (int loop = 0; loop < numOfquery; loop++)
	{
		string document;
		string retrieve_doc;
		getline(cin, document); // 整行讀取直到\n
		doc.push_back(document);
		getline(cin, retrieve_doc);
		ret_doc.push_back(retrieve_doc);
	}
	return numOfquery;
}
bool inRelevantDoc(int &num ,string& doc_id, vector<string>& rel_doc) /*judge 不給用 find() */
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
	double length = 1; // 計算走到第幾篇
	double doc_count = 0; // 計算預測的文章出現在相關文件的數量
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
	sum /= query_size; /*最後結果是相關文章的篇數，不是檢索後的文章有幾篇*/
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