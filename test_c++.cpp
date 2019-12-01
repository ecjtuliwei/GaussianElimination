#include<iostream>
#include "vector.h"
#include "matrix.h"
#include<ctime>
clock_t start, end_time;
using namespace std;
constexpr auto N = 1000;

void gaussin(matrix<double> a) {
	vector<double> c(N),x(N);
	for (int k = 1; k <= N; k++) {
		for (int i = k + 1; i <= N; i++) c[i-1] = a(i, k) / a(k, k);
		for (int i = k + 1; i <= N; i++)
			for (int j = 1; j <= N+1; j++)
				a(i , j ) = a(i , j ) - c[i-1] * a(k , j);
	}
	x[N - 1] = a(N ,N + 1) / a(N, N);
	for (int i = N - 2; i > -1; i--) {
		double sum = 0.0;
		for (int j = i + 1; j < N; j++) {
			sum += a(i + 1, j + 1) * x[j];
			x[i] = (a(i + 1, N + 1) - sum) / a(i + 1,i+1);
		}
	}
}
int main() {
	matrix<double> a;
	a.init(N, N+1);
	for (int i = 0; i < N*N+1; i++) {
			a[i] = rand();
	}
	start = clock();
	gaussin(a);
	end_time = clock();
	a.clear();
	double endtime = (float)(end_time - start) / CLOCKS_PER_SEC;
	cout << N << '*' << N << "的方程组解用时为：\n"<< endtime <<'s'<< endl;
	system("pause");
	return 0;
}