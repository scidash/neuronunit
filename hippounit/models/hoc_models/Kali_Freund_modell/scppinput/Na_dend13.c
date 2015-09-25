/* Created by Language version: 6.2.0 */
/* VECTORIZED */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scoplib.h"
#undef PI
 
#include "md1redef.h"
#include "section.h"
#include "md2redef.h"

#if METHOD3
extern int _method3;
#endif

#undef exp
#define exp hoc_Exp
extern double hoc_Exp();
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define gmax _p[0]
#define gion _p[1]
#define Xinf _p[2]
#define Xtau _p[3]
#define Yinf _p[4]
#define Ytau _p[5]
#define Zinf _p[6]
#define Ztau _p[7]
#define X _p[8]
#define Y _p[9]
#define Z _p[10]
#define ena _p[11]
#define ina _p[12]
#define DX _p[13]
#define DY _p[14]
#define DZ _p[15]
#define v _p[16]
#define _g _p[17]
#define _ion_ena	*_ppvar[0]._pval
#define _ion_ina	*_ppvar[1]._pval
#define _ion_dinadv	*_ppvar[2]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static int _hoc_rates();
 static int _mechtype;
extern int nrn_get_mechtype();
 static _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range();
 _prop = hoc_getdata_range(_mechtype);
 _extcall_prop = _prop;
 ret(1.);
}
 /* connect user functions to hoc names */
 static IntFunc hoc_intfunc[] = {
 "setdata_Na_dend13", _hoc_setdata,
 "rates_Na_dend13", _hoc_rates,
 0, 0
};
 
static void _check_rates(double*, Datum*, Datum*, _NrnThread*); 
static void _check_table_thread(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, int _type) {
   _check_rates(_p, _ppvar, _thread, _nt);
 }
 /* declare global and static user variables */
#define usetable usetable_Na_dend13
 double usetable = 1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "usetable_Na_dend13", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "gmax_Na_dend13", "S/cm2",
 "gion_Na_dend13", "S/cm2",
 "Xtau_Na_dend13", "ms",
 "Ytau_Na_dend13", "ms",
 "Ztau_Na_dend13", "ms",
 0,0
};
 static double X0 = 0;
 static double Y0 = 0;
 static double Z0 = 0;
 static double delta_t = 0.01;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "usetable_Na_dend13", &usetable_Na_dend13,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(), nrn_init(), nrn_state();
 static void nrn_cur(), nrn_jacob();
 
static int _ode_count(), _ode_map(), _ode_spec(), _ode_matsol();
 
#define _cvode_ieq _ppvar[3]._i
 /* connect range variables in _p that hoc is supposed to know about */
 static char *_mechanism[] = {
 "6.2.0",
"Na_dend13",
 "gmax_Na_dend13",
 0,
 "gion_Na_dend13",
 "Xinf_Na_dend13",
 "Xtau_Na_dend13",
 "Yinf_Na_dend13",
 "Ytau_Na_dend13",
 "Zinf_Na_dend13",
 "Ztau_Na_dend13",
 0,
 "X_Na_dend13",
 "Y_Na_dend13",
 "Z_Na_dend13",
 0,
 0};
 static Symbol* _na_sym;
 
static void nrn_alloc(_prop)
	Prop *_prop;
{
	Prop *prop_ion, *need_memb();
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 18, _prop);
 	/*initialize range parameters*/
 	gmax = 0.05;
 	_prop->param = _p;
 	_prop->param_size = 18;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_na_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ena */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ina */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dinadv */
 
}
 static _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 _Na_dend13_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("na", 1.0);
 	_na_sym = hoc_lookup("na_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
     _nrn_thread_table_reg(_mechtype, _check_table_thread);
  hoc_register_dparam_size(_mechtype, 4);
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 Na_dend13 /cygdrive/c/Documents and Settings/Saci/Desktop/NEURON 7.2/scppinput/Na_dend13.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double *_t_Xinf;
 static double *_t_Xtau;
 static double *_t_Yinf;
 static double *_t_Ytau;
 static double *_t_Zinf;
 static double *_t_Ztau;
static int _reset;
static char *modelname = "Channel: Na_dend";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static _modl_cleanup(){ _match_recurse=1;}
static _f_rates();
static rates();
 
static int _ode_spec1(), _ode_matsol1();
 static _n_rates();
 static int _slist1[3], _dlist1[3];
 static int states();
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {int _reset = 0; {
   rates ( _threadargscomma_ v ) ;
   DX = ( Xinf - X ) / Xtau ;
   DY = ( Yinf - Y ) / Ytau ;
   DZ = ( Zinf - Z ) / Ztau ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
 rates ( _threadargscomma_ v ) ;
 DX = DX  / (1. - dt*( ( ( ( - 1.0 ) ) ) / Xtau )) ;
 DY = DY  / (1. - dt*( ( ( ( - 1.0 ) ) ) / Ytau )) ;
 DZ = DZ  / (1. - dt*( ( ( ( - 1.0 ) ) ) / Ztau )) ;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) { {
   rates ( _threadargscomma_ v ) ;
    X = X + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / Xtau)))*(- ( ( ( Xinf ) ) / Xtau ) / ( ( ( ( - 1.0) ) ) / Xtau ) - X) ;
    Y = Y + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / Ytau)))*(- ( ( ( Yinf ) ) / Ytau ) / ( ( ( ( - 1.0) ) ) / Ytau ) - Y) ;
    Z = Z + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / Ztau)))*(- ( ( ( Zinf ) ) / Ztau ) / ( ( ( ( - 1.0) ) ) / Ztau ) - Z) ;
   }
  return 0;
}
 static double _mfac_rates, _tmin_rates;
  static void _check_rates(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  static double _sav_celsius;
  if (!usetable) {return;}
  if (_sav_celsius != celsius) { _maktable = 1;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_rates =  - 100.0 ;
   _tmax =  50.0 ;
   _dx = (_tmax - _tmin_rates)/3000.; _mfac_rates = 1./_dx;
   for (_i=0, _x=_tmin_rates; _i < 3001; _x += _dx, _i++) {
    _f_rates(_p, _ppvar, _thread, _nt, _x);
    _t_Xinf[_i] = Xinf;
    _t_Xtau[_i] = Xtau;
    _t_Yinf[_i] = Yinf;
    _t_Ytau[_i] = Ytau;
    _t_Zinf[_i] = Zinf;
    _t_Ztau[_i] = Ztau;
   }
   _sav_celsius = celsius;
  }
 }

 static rates(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _lv) { 
#if 0
_check_rates(_p, _ppvar, _thread, _nt);
#endif
 _n_rates(_p, _ppvar, _thread, _nt, _lv);
 return;
 }

 static _n_rates(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 _f_rates(_p, _ppvar, _thread, _nt, _lv); return; 
}
 _xi = _mfac_rates * (_lv - _tmin_rates);
 _i = (int) _xi;
 if (_xi <= 0.) {
 Xinf = _t_Xinf[0];
 Xtau = _t_Xtau[0];
 Yinf = _t_Yinf[0];
 Ytau = _t_Ytau[0];
 Zinf = _t_Zinf[0];
 Ztau = _t_Ztau[0];
 return; }
 if (_i >= 3000) {
 Xinf = _t_Xinf[3000];
 Xtau = _t_Xtau[3000];
 Yinf = _t_Yinf[3000];
 Ytau = _t_Ytau[3000];
 Zinf = _t_Zinf[3000];
 Ztau = _t_Ztau[3000];
 return; }
 _theta = _xi - (double)_i;
 Xinf = _t_Xinf[_i] + _theta*(_t_Xinf[_i+1] - _t_Xinf[_i]);
 Xtau = _t_Xtau[_i] + _theta*(_t_Xtau[_i+1] - _t_Xtau[_i]);
 Yinf = _t_Yinf[_i] + _theta*(_t_Yinf[_i+1] - _t_Yinf[_i]);
 Ytau = _t_Ytau[_i] + _theta*(_t_Ytau[_i+1] - _t_Ytau[_i]);
 Zinf = _t_Zinf[_i] + _theta*(_t_Zinf[_i+1] - _t_Zinf[_i]);
 Ztau = _t_Ztau[_i] + _theta*(_t_Ztau[_i+1] - _t_Ztau[_i]);
 }

 
static int  _f_rates ( _p, _ppvar, _thread, _nt, _lv ) double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt; 
	double _lv ;
 {
   double _lalpha , _lbeta , _ltau , _linf , _lgamma , _lzeta , _ltemp_adj_X , _lA_alpha_X , _lB_alpha_X , _lVhalf_alpha_X , _lA_beta_X , _lB_beta_X , _lVhalf_beta_X , _ltemp_adj_Y , _lA_tau_Y , _lB_tau_Y , _lVhalf_tau_Y , _lA_inf_Y , _lB_inf_Y , _lVhalf_inf_Y , _ltemp_adj_Z , _lA_alpha_Z , _lB_alpha_Z , _lVhalf_alpha_Z , _lA_beta_Z , _lB_beta_Z , _lVhalf_beta_Z ;
  _ltemp_adj_X = 1.0 ;
   _ltemp_adj_Y = 1.0 ;
   _ltemp_adj_Z = 1.0 ;
   _lA_alpha_X = 20000.0 ;
   _lB_alpha_X = 0.01 ;
   _lVhalf_alpha_X = - 0.03 ;
   _lA_alpha_X = _lA_alpha_X * 0.0010 ;
   _lB_alpha_X = _lB_alpha_X * 1000.0 ;
   _lVhalf_alpha_X = _lVhalf_alpha_X * 1000.0 ;
   _lalpha = _lA_alpha_X * exp ( ( _lv - _lVhalf_alpha_X ) / _lB_alpha_X ) ;
   _lA_beta_X = 20000.0 ;
   _lB_beta_X = - 0.00818182 ;
   _lVhalf_beta_X = - 0.03 ;
   _lA_beta_X = _lA_beta_X * 0.0010 ;
   _lB_beta_X = _lB_beta_X * 1000.0 ;
   _lVhalf_beta_X = _lVhalf_beta_X * 1000.0 ;
   _lbeta = _lA_beta_X * exp ( ( _lv - _lVhalf_beta_X ) / _lB_beta_X ) ;
   Xtau = 1.0 / ( _ltemp_adj_X * ( _lalpha + _lbeta ) ) ;
   Xinf = _lalpha / ( _lalpha + _lbeta ) ;
   _lv = _lv * 0.0010 ;
   _ltau = ( 1.0 / ( ( 300.0 * ( exp ( 0.2 * ( _lv + 0.036 ) / ( - 0.005 ) ) ) ) + ( 300.0 * ( exp ( ( 0.2 - 1.0 ) * ( _lv + 0.036 ) / ( - 0.005 ) ) ) ) ) + 0.0001 ) ;
   _ltau = _ltau * 1000.0 ;
   _lv = _lv * 1000.0 ;
   Ytau = _ltau / _ltemp_adj_Y ;
   _lv = _lv * 0.0010 ;
   _linf = 1.0 / ( 1.0 + exp ( - ( _lv + 0.036 ) / ( - 0.003 ) ) ) ;
   _lv = _lv * 1000.0 ;
   Yinf = _linf ;
   _lv = _lv * 0.0010 ;
   _lalpha = ( 1.0 + 0.3 * ( exp ( ( _lv + 0.03 ) / 0.002 ) ) ) / ( 1.0 + ( exp ( ( _lv + 0.03 ) / 0.002 ) ) ) * ( 1.0 + ( exp ( 450.0 * ( _lv + 0.045 ) ) ) ) / ( 5.0 * ( exp ( 90.0 * ( _lv + 0.045 ) ) ) + 0.002 * ( exp ( 450.0 * ( _lv + 0.045 ) ) ) ) ;
   _lalpha = _lalpha * 0.0010 ;
   _lbeta = ( 1.0 + ( exp ( 450.0 * ( _lv + 0.045 ) ) ) ) / ( 5.0 * ( exp ( 90.0 * ( _lv + 0.045 ) ) ) + 0.002 * ( exp ( 450.0 * ( _lv + 0.045 ) ) ) ) - ( 1.0 + 0.3 * ( exp ( ( _lv + 0.03 ) / 0.002 ) ) ) / ( 1.0 + ( exp ( ( _lv + 0.03 ) / 0.002 ) ) ) * ( 1.0 + ( exp ( 450.0 * ( _lv + 0.045 ) ) ) ) / ( 5.0 * ( exp ( 90.0 * ( _lv + 0.045 ) ) ) + 0.002 * ( exp ( 450.0 * ( _lv + 0.045 ) ) ) ) ;
   _lbeta = _lbeta * 0.0010 ;
   _lv = _lv * 1000.0 ;
   Ztau = 1.0 / ( _ltemp_adj_Z * ( _lalpha + _lbeta ) ) ;
   Zinf = _lalpha / ( _lalpha + _lbeta ) ;
    return 0; }
 
static int _hoc_rates() {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 
#if 1
 _check_rates(_p, _ppvar, _thread, _nt);
#endif
 _r = 1.;
 rates ( _p, _ppvar, _thread, _nt, *getarg(1) ) ;
 ret(_r);
}
 
static int _ode_count(_type) int _type;{ return 3;}
 
static int _ode_spec(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ena = _ion_ena;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
  }}
 
static int _ode_map(_ieq, _pv, _pvdot, _pp, _ppd, _atol, _type) int _ieq, _type; double** _pv, **_pvdot, *_pp, *_atol; Datum* _ppd; { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 3; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static int _ode_matsol(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ena = _ion_ena;
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_na_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_na_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_na_sym, _ppvar, 2, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
  int _i; double _save;{
  X = X0;
  Y = Y0;
  Z = Z0;
 {
   ena = 55.0 ;
   rates ( _threadargscomma_ v ) ;
   X = Xinf ;
   Y = Yinf ;
   Z = Zinf ;
   }
 
}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];

#if 0
 _check_rates(_p, _ppvar, _thread, _nt);
#endif
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
  ena = _ion_ena;
 initmodel(_p, _ppvar, _thread, _nt);
 }}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   gion = gmax * ( pow( ( X ) , 3.0 ) ) * ( pow( ( Y ) , 1.0 ) ) * ( pow( ( Z ) , 1.0 ) ) ;
   ina = gion * ( v - ena ) ;
   }
 _current += ina;

} return _current;
}

static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
  ena = _ion_ena;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dina;
  _dina = ina;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dinadv += (_dina - ina)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ina += ina ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}}

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type) {
 double _break, _save;
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _break = t + .5*dt; _save = t;
 v=_v;
{
  ena = _ion_ena;
 { {
 for (; t < _break; t += dt) {
   states(_p, _ppvar, _thread, _nt);
  
}}
 t = _save;
 } }}

}

static terminal(){}

static _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = &(X) - _p;  _dlist1[0] = &(DX) - _p;
 _slist1[1] = &(Y) - _p;  _dlist1[1] = &(DY) - _p;
 _slist1[2] = &(Z) - _p;  _dlist1[2] = &(DZ) - _p;
   _t_Xinf = makevector(3001*sizeof(double));
   _t_Xtau = makevector(3001*sizeof(double));
   _t_Yinf = makevector(3001*sizeof(double));
   _t_Ytau = makevector(3001*sizeof(double));
   _t_Zinf = makevector(3001*sizeof(double));
   _t_Ztau = makevector(3001*sizeof(double));
_first = 0;
}
