
#include <algorithm>
#include <iostream>
#include <cmath>
#include <ctime>
#include <complex>
#include <vector>
#include <set>
#include <list>


using namespace std;


//################################################################################
// 						     	Utilities
//################################################################################

int delta(int i, int j)
{
	if(i==j)
		return 1;
	else
		return 0;
}

double sinc(double x)
{
	if(x==0.)
		return 1;
	else
		return sin(M_PI*x) / (M_PI*x);
}
    
int argmax(vector<int>& v)
{
    return distance(v.begin(), max_element(v.begin(), v.end()));
}

void coordinates(int iPoint, int* Nd, int d, vector<int>& coords)
{
	coords[d-1] = iPoint;
	for(int i=0; i<d-1; ++i)
	{
		coords[i]   = coords[d-1] % Nd[i];
		coords[d-1] = coords[d-1] / Nd[i];
	}
	coords[d-1] = coords[d-1] % Nd[d-1];
}

void coordinates_blk(int iPoint, int* Index_begin, int* Index_end, int d, vector<int>& coords)
{
	int Ni;
	coords[d-1] = iPoint;
	for(int i=0; i<d; ++i)
	{
		Ni = Index_end[d-1-i] - Index_begin[d-1-i];
		coords[i]   = coords[d-1] % Ni + Index_begin[d-1-i];
		if(i==d-1) break;
		coords[d-1] = coords[d-1] / Ni;
	}
}

// void coordinates_blk(int iPoint, int* Index_begin, int* Index_end, int d, vector<int>& coords)
// {
// 	int Ni;
// 	coords[0] = iPoint;
// 	for(int i=d-1; i>=0; --i)
// 	{
// 		Ni = Index_end[i] - Index_begin[i];
// 		coords[i]   = coords[0] % Ni + Index_begin[i];
// 		if(i==0) break;
// 		coords[0] = coords[0] / Ni;
// 	}
// }



//################################################################################
// 						4th order Green operator
//################################################################################

double localValue_GreenHat(const vector<double>& k, int i, int j, int h, int l, double coef)
{
	double g1 = (delta(i,h)*k[j]*k[l] + delta(i,l)*k[j]*k[h] + delta(j,h)*k[i]*k[l] + delta(j,l)*k[i]*k[h])/4;
	double g2 = k[i]*k[j]*k[h]*k[l];
	// return g1 - (lambda+mu)/(lambda+2*mu) * g2; // 1/mu is ommited !
	return g1 - g2 * coef; // 1/mu is ommited !
}


void Voigt2Index(int I, int d, int &i, int &j)
{
	if(I<d)
	{
		i = I;
		j = I;
	}
	else if(I==d)
	{
		i = d-2;
		j = d-1;
	}
	else if(I==d+1)
	{
		i = d-3;
		j = d-1;
	}
	else if(I==d+2)
	{
		i = d-3;
		j = d-2;
	}
}


// // Construct Fourier transform of Green operator
// void construct_GreenHat(double* G, int* Index_begin, int* Index_end, int d, double lambda, double mu, double* frq, int Ntrunc)
// {
// 	const int nVoigt = d*(d+1)/2;
// 	const int nVoigt2 = nVoigt*nVoigt;
// 	const double coef = (lambda+mu)/(lambda+2*mu);

// 	int i, j, h, l;
// 	int I, J;
// 	int mm;
// 	int p;

// 	double fct0=1., fct=1.;
// 	vector<int> a(d, 0), m(d, -Ntrunc);
// 	vector<double> f(d), k(d);
// 	double norm_k = 0.;
// 	double g;
// 	bool cond;

// 	vector<int> Nd(d);
// 	for(i=0; i<d; ++i) Nd[i] = Index_end[i]-Index_begin[i];

// 	int nPoints = 1;
// 	for(i=0; i<d; ++i) nPoints *= Nd[i];

// 	int MMM = pow(2*Ntrunc+1, d);

// 	int Nmin = *min_element(Index_begin, Index_begin+d);
// 	int Nmax = *max_element(Index_end, Index_end+d);
// 	vector<double> sin_coef(Nmax);
// 	for(i=Nmin; i<Nmax; ++i) sin_coef[i] = sin(M_PI*frq[i])/M_PI;
	
// 	for(int iPoint=0; iPoint<nPoints; ++iPoint)
// 	{
// 		p = iPoint*nVoigt2;		
// 		// coordinates(iPoint, Nd, d, a);
// 		coordinates_blk(iPoint, Index_begin, Index_end, d, a);

// 		if(iPoint!=0)
// 		{

// 			// for(i=0; i<d; ++i) f[i] = frq[a[i]];

// 			// if(Ntrunc!=0)
// 			{
// 				fct0 = 1.;
// 				for(i=0; i<d; ++i)
// 				{
// 					f[i] = frq[a[i]];
// 					if(f[i]!=0) fct0 *= sin_coef[a[i]];
// 					// if(f[i]!=0) fct0 *= sin(M_PI*f[i])/M_PI;
// 				}
// 				fct0 *= fct0;
// 			}

// 			m[d-1] = -Ntrunc;
// 			for(mm=0; mm<MMM; ++mm)
// 			{
// 				cond = false;
// 				for(i=0; i<d; ++i) cond = cond || (a[i]==0 && m[i]!=0);

// 				if(!cond)
// 				{
// 					for(i=0; i<d; ++i) k[i] = f[i] + m[i];

// 					// if(Ntrunc!=0)
// 					{
// 						fct = 1.;
// 						for(i=0; i<d; ++i) if(f[i]!=0) fct /= k[i];
// 						fct *= fct*fct0;
// 					}

// 					// if(Ntrunc!=0)
// 					// {
// 					// 	fct1 = 1.;
// 					// 	for(i=0; i<d; ++i) fct1 *= sinc(k[i]);
// 					// 	fct1 *= fct1;
// 					// }
// 					// if(a[0]==0 && m[0]!=0 && fct1>1.e-10) cout << fct1 << endl;
// 					// if(abs(fct1-fct) > 1.e-10) cout << a[0]*a[1] << "  " << abs(fct1-fct) << endl;
// 					// fct = fct1;


// 					// if(fct>1.e-5)
// 					{
// 						norm_k = 0.;
// 						for(i=0; i<d; ++i) norm_k += k[i]*k[i];
// 						norm_k = sqrt(norm_k);

// 						if(norm_k != 0.)
// 						{
// 							for(i=0; i<d; ++i) k[i] /= norm_k;

// 							for(I=0; I<nVoigt; ++I)
// 								for(J=0; J<nVoigt; ++J)
// 								{
// 									Voigt2Index(I,d,i,j);
// 									Voigt2Index(J,d,h,l);
// 									g = localValue_GreenHat(k, i, j, h, l, coef);
// 									if(J>=d) g *= 2;
// 									G[p + I*nVoigt + J] += fct * g;
// 								}
// 						}
// 					}
// 				}

// 				for(i=0; i<d; ++i) if(++m[i] > Ntrunc) m[i] = -Ntrunc; else break;
// 			}
// 		}
		
// 		// for(i=0; i<d; ++i) if(++a[i] >= Nd[i]) a[i] = 0; else break;

// 		// ++a[0];
// 		// for(i=0; i<d-1; ++i)
// 		// {
// 		// 	if(a[i] == Nd[i])
// 		// 	{
// 		// 		a[i] = 0;
// 		// 		++a[i+1];
// 		// 	}
// 		// 	else break;
// 		// }

// 	}
// }


// Construct Fourier transform of Green operator (filtered inconsistent)
void construct_GreenHat(double* Gamma, int* Index_begin, int* Index_end, int d, double lambda, double mu, double* frq, int Ntrunc)
{
	const int nVoigt = d*(d+1)/2;
	const int nVoigt2 = nVoigt*nVoigt;
	const double coef = (lambda+mu)/(lambda+2*mu);

	int i, j, h, l;
	int I, J;
	int mm;
	int p;

	double fct=1.;
	vector<int> a(d, 0), m(d, -1);
	vector<double> f(d), k(d);
	double norm_k = 0.;
	double g;

	vector<int> Nd(d);
	for(i=0; i<d; ++i) Nd[i] = Index_end[i]-Index_begin[i];

	int nPoints = 1;
	for(i=0; i<d; ++i) nPoints *= Nd[i];

	int MMM = pow(2, d);
	
	for(int iPoint=0; iPoint<nPoints; ++iPoint)
	{
		p = iPoint*nVoigt2;		
		// coordinates(iPoint, Nd, d, a);
		coordinates_blk(iPoint, Index_begin, Index_end, d, a);

		for(i=0; i<d; ++i) f[i] = frq[a[i]];

		if(iPoint!=0)
		{

			for(i=0; i<d; ++i) m[i] = -1;

			for(mm=0; mm<MMM; ++mm)
			{
				for(i=0; i<d; ++i) k[i] = f[i] + m[i];

				fct = 1.;
				for(i=0; i<d; ++i) fct *= cos(0.5*M_PI*k[i]);
				fct *= fct;

				norm_k = 0.;
				for(i=0; i<d; ++i) norm_k += k[i]*k[i];
				norm_k = sqrt(norm_k);

				if(norm_k != 0.)
				{
					for(i=0; i<d; ++i) k[i] /= norm_k;

					for(I=0; I<nVoigt; ++I)
						for(J=0; J<nVoigt; ++J)
						{
							Voigt2Index(I,d,i,j);
							Voigt2Index(J,d,h,l);
							g = localValue_GreenHat(k, i, j, h, l, coef);
							if(J>=d) g *= 2;
							Gamma[p + I*nVoigt + J] += fct * g;
						}
				}

				for(i=0; i<d; ++i) if(++m[i] > 0) m[i] = -1; else break;
			}
		}

	}
}




complex<double> gen(double* G, int n, complex<double>* tau_hat, int& pk)
{
	// static int pk = 0;
	int k = pk % n;
	int p = pk - k;
	complex<double> entry = 0;
	for(int l=0; l<n; ++l)
		entry += G[(p + k)*n + l] * tau_hat[p + l];
	++pk;
	return entry;
}


// Apply Green operator
void apply_GreenOperator(double* G, int* Nd, int n, complex<double>* tau_hat, complex<double>* eta_hat)
{
	// clock_t begin = clock();

	int d = (int)(sqrt(8.*n+1.)-1.)/2;

	int i, k, l, p, iPoint;
	complex<double> entry;

	int nPoints = 1;
	for(i=0; i<d; ++i) nPoints *= Nd[i];
	// const int size = nPoints*n;

// #pragma omp parallel for num_threads(NUM_THREADS)
// #pragma omp simd
// #pragma omp parallel for
	for(iPoint=0; iPoint<nPoints; ++iPoint)	
	{
		// int cpu_num = sched_getcpu();
		// cout << cpu_num;
		p = iPoint*n;
		for(k=0; k<n; ++k)
		{
			entry = 0;
			for(l=0; l<n; ++l)
				entry += G[(p + k)*n + l] * tau_hat[p + l];
			eta_hat[p + k] = entry;
		}
	}

	// clock_t end = clock();
  	// double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	// cout << "Green_cpp time: " << elapsed_secs << endl;
}


// Apply Matrix
void apply_Matrix(int* Phase, double* C1, double* C2, int* Nd, int n, double* x, double* y)
{
	int d = (int)(sqrt(8.*n+1.)-1.)/2;

	int i, k, l, p, iPoint;
	double entry;

	int nPoints = 1;
	for(i=0; i<d; ++i) nPoints *= Nd[i];
	// for(int i=0; i<nPoints*n; ++i) y[i] = 0.;


// #pragma omp parallel for num_threads(NUM_THREADS)
// #pragma omp simd
// #pragma omp parallel for
	for(iPoint=0; iPoint<nPoints; ++iPoint)
	{
		p = iPoint*n;
		if(Phase[iPoint])
			for(k=0; k<n; ++k)
			{
				entry = 0;
				for(l=0; l<n; ++l)
					entry += C1[k*n + l] * x[p + l];
				y[p + k] = entry;
			}
		else
			for(k=0; k<n; ++k)
			{
				entry = 0;
				for(l=0; l<n; ++l)
					entry += C2[k*n + l] * x[p + l];
				y[p + k] = entry;
			}
	}

}








//################################################################################
// 						   Largest inclusion search
//################################################################################


//-------------------------------------------------------------------------------
// Inclusions indexing
//-------------------------------------------------------------------------------

class InclusionSearcher
{
	// Attributes
	public:
		int d;
		int* Nd;
		int nVoxels;
		int* Phase;

		int* indexIncl;
		int nIncl;
		vector<int> volume;
		vector<double> surface;

		int* markerLargestIncl;

	private:
		set<int> TreatedVoxels, CurrentInclusion;
		list<int> InclusionStack;
		vector<int> coords, a;
		int nCube, iCenter, depth_max;


	// Methods
	public:
		InclusionSearcher(int* p_Phase, int* p_Nd, int p_d, int* p_indexIncl)
		: d(p_d), Nd(p_Nd), Phase(p_Phase), indexIncl(p_indexIncl), nIncl(0)
		{
			nVoxels = 1;
			for(int i=0; i<d; ++i) nVoxels *= Nd[i];

			coords.assign(d,0);
			nCube = pow(3, d);
			iCenter = (int)(nCube-1)/2;
		}

		// int getAllInclusions()
		// {
		// 	fill(indexIncl, indexIncl+nVoxels, 0);
		// 	pair<int,int> vol_srf(0,0);

		// 	// iterate over all voxels
		// 	for(int iVoxel=0; iVoxel<nVoxels; ++iVoxel)
		// 	{
		// 		if(!isTreated(iVoxel))
		// 		{
		// 			CurrentInclusion.clear();
		// 			vol_srf = getInclusion(iVoxel);

		// 			if(!CurrentInclusion.empty())
		// 			{
		// 				++nIncl;
		// 				volume.push_back(vol_srf.first);
		// 				surface.push_back(vol_srf.second);
		// 				for (set<int>::iterator it=CurrentInclusion.begin(); it!=CurrentInclusion.end(); ++it) indexIncl[*it] = nIncl;
		// 				TreatedVoxels.insert(CurrentInclusion.begin(), CurrentInclusion.end());
		// 			}
		// 		} // end if
		// 	} // end for

		// 	return nIncl;
		// }

		int getAllInclusions()
		{
			fill(indexIncl, indexIncl+nVoxels, 0);
			TreatedVoxels.clear();
			volume.clear();
			surface.clear();
			nIncl = 0;

			// iterate over all voxels
			for(int iVoxel=0; iVoxel<nVoxels; ++iVoxel)
			{
				getInclusion(iVoxel);
			}

			return nIncl;
		}

		int getLargestInclusion(int* p_markerLargestIncl)
		{
			markerLargestIncl = p_markerLargestIncl;
			fill(markerLargestIncl, markerLargestIncl+nVoxels, 0);

			int iLargest = argmax(volume)+1;

			for(int iVoxel=0; iVoxel<nVoxels; ++iVoxel)
				if(indexIncl[iVoxel] == iLargest) markerLargestIncl[iVoxel] = 1;

			return iLargest;
		}

		int getSubdomainVicinity(int* p_Subdomain, int p_VicinityDepth, bool vicinity_only=false)
		{
			depth_max = p_VicinityDepth;
			int* subdomain = p_Subdomain;
			int iVoxel, i, n, vol = 0;

			list<int> SubdomainStack;
			for(iVoxel=0; iVoxel<nVoxels; ++iVoxel) if(subdomain[iVoxel]) SubdomainStack.push_back(iVoxel);
			list<int> InitSubdomainStack(SubdomainStack);

			vector<int> Neighbours;
			vector<int>::iterator iNeighbour;

			for(int depth=0; depth<depth_max; ++depth)
			{
				n = SubdomainStack.size();
				vol += n;
				for(i=0; i<n; ++i)
				{
					iVoxel = SubdomainStack.front();
					SubdomainStack.pop_front();
					getNeighbours(iVoxel, Neighbours);
					for(iNeighbour=Neighbours.begin(); iNeighbour!=Neighbours.end(); ++iNeighbour)
						if(!subdomain[*iNeighbour])
						{
							subdomain[*iNeighbour] = 1;
							SubdomainStack.push_back(*iNeighbour);
						}
				}

			}

			if(vicinity_only)
			{
				n = SubdomainStack.size();
				for(i=0; i<n; ++i)
				{
					iVoxel = InitSubdomainStack.front();
					SubdomainStack.pop_front();
					subdomain[iVoxel] = 0;
				}
			}

			return vol;
		}

	private:

		bool isTreated(int iVoxel)
		{
			return TreatedVoxels.find(iVoxel) != TreatedVoxels.end();
		}

		void getNeighbours(int iVoxel, vector<int>& Neighbours)
		{
			Neighbours.clear();
			a.assign(d,-1);
			coordinates(iVoxel, Nd, d, coords);
			int i, iNeighbourVoxel;
			for(int k=0; k<nCube; ++k)
			{
				if(k!=iCenter)
				{
					iNeighbourVoxel = 0;
					for(i=d-1; i>=0; --i)
					{
						iNeighbourVoxel += (coords[i] + a[i] + Nd[i]) % Nd[i];
						if(i) iNeighbourVoxel *= Nd[i-1];
					}
					Neighbours.push_back(iNeighbourVoxel);
				}
				for(i=0; i<d; ++i) if(++a[i] > 1) a[i] = -1; else break;
			}
		}

		int pushNeighbours(int iVoxel, list<int>& Stack)
		{
			a.assign(d,-1);
			coordinates(iVoxel, Nd, d, coords);
			int i, iNeighbourVoxel, norm;
			int nOuterNeighbours = 0;
			for(int k=0; k<nCube; ++k)
			{
				norm = 0;
				for(i=0; i<d; ++i) norm = norm + a[i]*a[i];

				if(norm==1)
				{
					iNeighbourVoxel = 0;
					for(i=d-1; i>=0; --i)
					{
						iNeighbourVoxel += (coords[i] + a[i] + Nd[i]) % Nd[i];
						if(i) iNeighbourVoxel *= Nd[i-1];
					}

					if(Phase[iNeighbourVoxel]) Stack.push_back(iNeighbourVoxel);
					else if(norm==1) ++nOuterNeighbours;
				}
				for(i=0; i<d; ++i) if(++a[i] > 1) a[i] = -1; else break;
			}
			return nOuterNeighbours;
		}

		// void getDirectNeighbours(int iVoxel, vector<int>& Neighbours)
		// {
		// 	Neighbours.clear();
		// 	a.assign(d,-1);
		// 	coordinates(iVoxel, Nd, d, coords);
		// 	int i, iNeighbourVoxel, norm;
		// 	for(int k=0; k<nCube; ++k)
		// 	{
		// 		norm = 0;
		// 		for(i=0; i<d; ++i) norm = norm + a[i]*a[i];
		// 		if(norm==1)
		// 		{
		// 			iNeighbourVoxel = 0;
		// 			for(i=d-1; i>=0; --i)
		// 			{
		// 				iNeighbourVoxel += (coords[i] + a[i] + Nd[i]) % Nd[i];
		// 				if(i) iNeighbourVoxel *= Nd[i-1];
		// 			}
		// 			Neighbours.push_back(iNeighbourVoxel);
		// 		}
		// 		for(i=0; i<d; ++i) if(++a[i] > 1) a[i] = -1; else break;
		// 	}
		// }

		// pair<int,int> getInclusion(int iVoxel)
		// {
		// 	pair<int,int> vol_srf(0,0);
		// 	pair<int,int> add_vol_srf;
		// 	const int n = nCube-1;
		// 	// bool inside = true;
		// 	if(Phase[iVoxel])
		// 		if(CurrentInclusion.insert(iVoxel).second)
		// 		{
		// 			vector<int> vNeighbour(n);
		// 			getNeighbours(iVoxel, vNeighbour);
		// 			cout << "vNeighbour : " << vNeighbour.size() << endl;
		// 			// int* Neighbours = new int[n];
		// 			// cout << "Neighbours : " << Neighbours << endl;
		// 			// vector<int>::iterator iNeighbour;
		// 			// getNeighbours(iVoxel, Neighbours);
		// 			for(int i=0; i<n; ++i)
		// 			{
		// 				// cout << *iNeighbour << "  " << nVoxels << endl;
		// 				// add_vol_srf = 
		// 				getInclusion(vNeighbour[i]);
		// 				// vol_srf.first  += add_vol_srf.first;
		// 				// vol_srf.second += add_vol_srf.second;
		// 			}
		// 			// delete Neighbours;
		// 			// for(iNeighbour=Neighbours.begin(); iNeighbour!=Neighbours.end(); ++iNeighbour)
		// 			// {
		// 			// 	cout << *iNeighbour << "  " << nVoxels << endl;
		// 			// 	// add_vol_srf = 
		// 			// 	getInclusion(*iNeighbour);
		// 			// 	// vol_srf.first  += add_vol_srf.first;
		// 			// 	// vol_srf.second += add_vol_srf.second;
		// 			// }
		// 			// getNeighbours(iVoxel);
		// 			// for(iNeighbour=Neighbours.begin(); iNeighbour!=Neighbours.end(); ++iNeighbour)
		// 			// 	if(!Phase[*iNeighbour]) ++vol_srf.second;
		// 			// if(!inside) ++vol_srf.second;
		// 			// ++(vol_srf.first);
		// 		}
		// 	return vol_srf;
		// }

		void getInclusion(int iInitialVoxel)
		{
			if(!isTreated(iInitialVoxel) && Phase[iInitialVoxel])
			{
				++nIncl;
				int vol = 0;
				double srf = 0.;
				double w = 0.;

				CurrentInclusion.clear();
				InclusionStack.clear();
				InclusionStack.push_back(iInitialVoxel);

				while(!InclusionStack.empty())
				{
					int iVoxel = InclusionStack.front();
					InclusionStack.pop_front();
					if(CurrentInclusion.insert(iVoxel).second)
					{
						++vol;
						int nOuterNeighbours = pushNeighbours(iVoxel, InclusionStack);
						// switch (nOuterNeighbours)
						// {
						// 	case 1: w = 1; break;
						// 	case 2: w = 1.5; break;
						// 	case 3: w = 2; break;
						// 	case 4: w = 3.5; break;
						// 	case 5: w = 4.2; break;
						// 	case 6: w = 4.9; break;		
						// 	default: w = 0.; break;					
						// }
						// srf = srf + w;
						srf = srf + nOuterNeighbours;
						// if(nOuterNeighbours>0) ++srf;
						indexIncl[iVoxel] = nIncl;
						TreatedVoxels.insert(iVoxel);
					}
				}

				volume.push_back(vol);
				surface.push_back(srf);
			}			
		}

}; // end class

int indexInclusions(int* Phase, int* Nd, int d, int* indexIncl)
{
	InclusionSearcher IS(Phase, Nd, d, indexIncl);	
	return IS.getAllInclusions();
}


int findLargestInclusion(int* Phase, int* Nd, int d, int* indexIncl, int* markerLargestIncl)
{
	InclusionSearcher IS(Phase, Nd, d, indexIncl);
	IS.getAllInclusions();
	return IS.getLargestInclusion(markerLargestIncl);
}


int findLargestInclusionVicinity(int* Phase, int* Nd, int d, int* indexIncl, int* markerLargestIncl, int vicinity_depth=0, bool vicinity_only=false)
{
	InclusionSearcher IS(Phase, Nd, d, indexIncl);
	IS.getAllInclusions();
	if(vicinity_depth)
	{
		IS.getLargestInclusion(markerLargestIncl);	
		return IS.getSubdomainVicinity(markerLargestIncl, vicinity_depth, vicinity_only);
	}
	else
		return IS.getLargestInclusion(markerLargestIncl);
}



int findAllInclusionsVolumesAndSurfaces(int* Phase, int* Nd, int d, int* indexIncl, int* volume, double* surface)
{
	InclusionSearcher IS(Phase, Nd, d, indexIncl);
	int nIncl = IS.getAllInclusions();
	
	for(int i=0; i<nIncl; ++i)
	{
		volume[i]  = IS.volume[i];
		surface[i] = IS.surface[i];
	}

	return nIncl;
}