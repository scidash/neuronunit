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
#define gcatbar _p[0]
#define iCa _p[1]
#define hinf _p[2]
#define tauh _p[3]
#define minf _p[4]
#define taum _p[5]
#define m _p[6]
#define h _p[7]
#define cai _p[8]
#define cao _p[9]
#define Dm _p[10]
#define Dh _p[11]
#define gcat _p[12]
#define v _p[13]
#define _g _p[14]
#define _ion_cai	*_ppvar[0]._pval
#define _ion_cao	*_ppvar[1]._pval
#define _ion_iCa	*_ppvar[2]._pval
#define _ion_diCadv	*_ppvar[3]._pval
 
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
 static int _hoc_KTF();
 static int _hoc_alpm();
 static int _hoc_alph();
 static int _hoc_betm();
 static int _hoc_beth();
 static int _hoc_efun();
 static int _hoc_ghk();
 static int _hoc_h2();
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
 "setdata_cat", _hoc_setdata,
 "KTF_cat", _hoc_KTF,
 "alpm_cat", _hoc_alpm,
 "alph_cat", _hoc_alph,
 "betm_cat", _hoc_betm,
 "beth_cat", _hoc_beth,
 "efun_cat", _hoc_efun,
 "ghk_cat", _hoc_ghk,
 "h2_cat", _hoc_h2,
 "rates_cat", _hoc_rates,
 0, 0
};
#define KTF KTF_cat
#define _f_betm _f_betm_cat
#define _f_alpm _f_alpm_cat
#define _f_beth _f_beth_cat
#define _f_alph _f_alph_cat
#define alpm alpm_cat
#define alph alph_cat
#define betm betm_cat
#define beth beth_cat
#define efun efun_cat
#define ghk ghk_cat
#define h2 h2_cat
 extern double KTF();
 extern double _f_betm();
 extern double _f_alpm();
 extern double _f_beth();
 extern double _f_alph();
 extern double alpm();
 extern double alph();
 extern double betm();
 extern double beth();
 extern double efun();
 extern double ghk();
 extern double h2();
 
static void _check_alph(double*, Datum*, Datum*, _NrnThread*); 
static void _check_beth(double*, Datum*, Datum*, _NrnThread*); 
static void _check_alpm(double*, Datum*, Datum*, _NrnThread*); 
static void _check_betm(double*, Datum*, Datum*, _NrnThread*); 
static void _check_table_thread(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, int _type) {
   _check_alph(_p, _ppvar, _thread, _nt);
   _check_beth(_p, _ppvar, _thread, _nt);
   _check_alpm(_p, _ppvar, _thread, _nt);
   _check_betm(_p, _ppvar, _thread, _nt);
 }
 /* declare global and static user variables */
#define eca eca_cat
 double eca = 140;
#define ki ki_cat
 double ki = 0.001;
#define tfi tfi_cat
 double tfi = 0.68;
#define tfa tfa_cat
 double tfa = 1;
#define tBase tBase_cat
 double tBase = 23.5;
#define usetable usetable_cat
 double usetable = 1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "usetable_cat", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "tBase_cat", "degC",
 "ki_cat", "mM",
 "gcatbar_cat", "mho/cm2",
 "iCa_cat", "mA/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double h0 = 0;
 static double m0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "tBase_cat", &tBase_cat,
 "ki_cat", &ki_cat,
 "tfa_cat", &tfa_cat,
 "tfi_cat", &tfi_cat,
 "eca_cat", &eca_cat,
 "usetable_cat", &usetable_cat,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(), nrn_init(), nrn_state();
 static void nrn_cur(), nrn_jacob();
 
static int _ode_count(), _ode_map(), _ode_spec(), _ode_matsol();
 
#define _cvode_ieq _ppvar[4]._i
 /* connect range variables in _p that hoc is supposed to know about */
 static char *_mechanism[] = {
 "6.2.0",
"cat",
 "gcatbar_cat",
 0,
 "iCa_cat",
 "hinf_cat",
 "tauh_cat",
 "minf_cat",
 "taum_cat",
 0,
 "m_cat",
 "h_cat",
 0,
 0};
 static Symbol* _ca_sym;
 static Symbol* _Ca_sym;
 
static void nrn_alloc(_prop)
	Prop *_prop;
{
	Prop *prop_ion, *need_memb();
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 15, _prop);
 	/*initialize range parameters*/
 	gcatbar = 0;
 	_prop->param = _p;
 	_prop->param_size = 15;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 5, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_ca_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[0]._pval = &prop_ion->param[1]; /* cai */
 	_ppvar[1]._pval = &prop_ion->param[2]; /* cao */
 prop_ion = need_memb(_Ca_sym);
 	_ppvar[2]._pval = &prop_ion->param[3]; /* iCa */
 	_ppvar[3]._pval = &prop_ion->param[4]; /* _ion_diCadv */
 
}
 static _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 _cat_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("ca", -10000.);
 	ion_reg("Ca", 2.0);
 	_ca_sym = hoc_lookup("ca_ion");
 	_Ca_sym = hoc_lookup("Ca_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
     _nrn_thread_table_reg(_mechtype, _check_table_thread);
  hoc_register_dparam_size(_mechtype, 5);
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 cat /cygdrive/c/Users/Saci/Dokumentumok/szakmai gyak/Ca1_Bianchi/Ca1_Bianchi/experiment/cat.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double FARADAY = 96520.0;
 static double R = 8.3134;
 static double KTOMV = .0853;
 static double *_t_alph;
 static double *_t_beth;
 static double *_t_alpm;
 static double *_t_betm;
static int _reset;
static char *modelname = "T-calcium channel";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static _modl_cleanup(){ _match_recurse=1;}
static rates();
 
static int _ode_spec1(), _ode_matsol1();
 static double _n_betm();
 static double _n_alpm();
 static double _n_beth();
 static double _n_alph();
 static int _slist1[2], _dlist1[2];
 static int states();
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {int _reset = 0; {
   rates ( _threadargscomma_ v ) ;
   Dm = ( minf - m ) / taum ;
   Dh = ( hinf - h ) / tauh ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
 rates ( _threadargscomma_ v ) ;
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / taum )) ;
 Dh = Dh  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tauh )) ;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) { {
   rates ( _threadargscomma_ v ) ;
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / taum)))*(- ( ( ( minf ) ) / taum ) / ( ( ( ( - 1.0) ) ) / taum ) - m) ;
    h = h + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tauh)))*(- ( ( ( hinf ) ) / tauh ) / ( ( ( ( - 1.0) ) ) / tauh ) - h) ;
   }
  return 0;
}
 
double h2 ( _p, _ppvar, _thread, _nt, _lcai ) double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt; 
	double _lcai ;
 {
   double _lh2;
 _lh2 = ki / ( ki + _lcai ) ;
   
return _lh2;
 }
 
static int _hoc_h2() {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  h2 ( _p, _ppvar, _thread, _nt, *getarg(1) ) ;
 ret(_r);
}
 
double ghk ( _p, _ppvar, _thread, _nt, _lv , _lci , _lco ) double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt; 
	double _lv , _lci , _lco ;
 {
   double _lghk;
 double _lnu , _lf ;
 _lf = KTF ( _threadargscomma_ celsius ) / 2.0 ;
   _lnu = _lv / _lf ;
   _lghk = - _lf * ( 1. - ( _lci / _lco ) * exp ( _lnu ) ) * efun ( _threadargscomma_ _lnu ) ;
   
return _lghk;
 }
 
static int _hoc_ghk() {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  ghk ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) , *getarg(3) ) ;
 ret(_r);
}
 
double KTF ( _p, _ppvar, _thread, _nt, _lcelsius ) double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt; 
	double _lcelsius ;
 {
   double _lKTF;
 _lKTF = ( ( 25. / 293.15 ) * ( _lcelsius + 273.15 ) ) ;
   
return _lKTF;
 }
 
static int _hoc_KTF() {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  KTF ( _p, _ppvar, _thread, _nt, *getarg(1) ) ;
 ret(_r);
}
 
double efun ( _p, _ppvar, _thread, _nt, _lz ) double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt; 
	double _lz ;
 {
   double _lefun;
 if ( fabs ( _lz ) < 1e-4 ) {
     _lefun = 1.0 - _lz / 2.0 ;
     }
   else {
     _lefun = _lz / ( exp ( _lz ) - 1.0 ) ;
     }
   
return _lefun;
 }
 
static int _hoc_efun() {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  efun ( _p, _ppvar, _thread, _nt, *getarg(1) ) ;
 ret(_r);
}
 static double _mfac_alph, _tmin_alph;
  static void _check_alph(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  if (!usetable) {return;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_alph =  - 150.0 ;
   _tmax =  150.0 ;
   _dx = (_tmax - _tmin_alph)/200.; _mfac_alph = 1./_dx;
   for (_i=0, _x=_tmin_alph; _i < 201; _x += _dx, _i++) {
    _t_alph[_i] = _f_alph(_p, _ppvar, _thread, _nt, _x);
   }
  }
 }

 double alph(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _lv) { 
#if 0
_check_alph(_p, _ppvar, _thread, _nt);
#endif
 return _n_alph(_p, _ppvar, _thread, _nt, _lv);
 }

 static double _n_alph(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 return _f_alph(_p, _ppvar, _thread, _nt, _lv); 
}
 _xi = _mfac_alph * (_lv - _tmin_alph);
 _i = (int) _xi;
 if (_xi <= 0.) {
 return _t_alph[0];
 }
 if (_i >= 200) {
 return _t_alph[200];
 }
 return _t_alph[_i] + (_xi - (double)_i)*(_t_alph[_i+1] - _t_alph[_i]);
 }

 
double _f_alph ( _p, _ppvar, _thread, _nt, _lv ) double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt; 
	double _lv ;
 {
   double _lalph;
 _lalph = 1.6e-4 * exp ( - ( _lv + 57.0 ) / 19.0 ) ;
   
return _lalph;
 }
 
static int _hoc_alph() {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 
#if 1
 _check_alph(_p, _ppvar, _thread, _nt);
#endif
 _r =  alph ( _p, _ppvar, _thread, _nt, *getarg(1) ) ;
 ret(_r);
}
 static double _mfac_beth, _tmin_beth;
  static void _check_beth(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  if (!usetable) {return;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_beth =  - 150.0 ;
   _tmax =  150.0 ;
   _dx = (_tmax - _tmin_beth)/200.; _mfac_beth = 1./_dx;
   for (_i=0, _x=_tmin_beth; _i < 201; _x += _dx, _i++) {
    _t_beth[_i] = _f_beth(_p, _ppvar, _thread, _nt, _x);
   }
  }
 }

 double beth(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _lv) { 
#if 0
_check_beth(_p, _ppvar, _thread, _nt);
#endif
 return _n_beth(_p, _ppvar, _thread, _nt, _lv);
 }

 static double _n_beth(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 return _f_beth(_p, _ppvar, _thread, _nt, _lv); 
}
 _xi = _mfac_beth * (_lv - _tmin_beth);
 _i = (int) _xi;
 if (_xi <= 0.) {
 return _t_beth[0];
 }
 if (_i >= 200) {
 return _t_beth[200];
 }
 return _t_beth[_i] + (_xi - (double)_i)*(_t_beth[_i+1] - _t_beth[_i]);
 }

 
double _f_beth ( _p, _ppvar, _thread, _nt, _lv ) double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt; 
	double _lv ;
 {
   double _lbeth;
 _lbeth = 1.0 / ( exp ( ( - _lv + 15.0 ) / 10.0 ) + 1.0 ) ;
   
return _lbeth;
 }
 
static int _hoc_beth() {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 
#if 1
 _check_beth(_p, _ppvar, _thread, _nt);
#endif
 _r =  beth ( _p, _ppvar, _thread, _nt, *getarg(1) ) ;
 ret(_r);
}
 static double _mfac_alpm, _tmin_alpm;
  static void _check_alpm(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  if (!usetable) {return;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_alpm =  - 150.0 ;
   _tmax =  150.0 ;
   _dx = (_tmax - _tmin_alpm)/200.; _mfac_alpm = 1./_dx;
   for (_i=0, _x=_tmin_alpm; _i < 201; _x += _dx, _i++) {
    _t_alpm[_i] = _f_alpm(_p, _ppvar, _thread, _nt, _x);
   }
  }
 }

 double alpm(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _lv) { 
#if 0
_check_alpm(_p, _ppvar, _thread, _nt);
#endif
 return _n_alpm(_p, _ppvar, _thread, _nt, _lv);
 }

 static double _n_alpm(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 return _f_alpm(_p, _ppvar, _thread, _nt, _lv); 
}
 _xi = _mfac_alpm * (_lv - _tmin_alpm);
 _i = (int) _xi;
 if (_xi <= 0.) {
 return _t_alpm[0];
 }
 if (_i >= 200) {
 return _t_alpm[200];
 }
 return _t_alpm[_i] + (_xi - (double)_i)*(_t_alpm[_i+1] - _t_alpm[_i]);
 }

 
double _f_alpm ( _p, _ppvar, _thread, _nt, _lv ) double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt; 
	double _lv ;
 {
   double _lalpm;
 _lalpm = 0.1967 * ( - 1.0 * _lv + 19.88 ) / ( exp ( ( - 1.0 * _lv + 19.88 ) / 10.0 ) - 1.0 ) ;
   
return _lalpm;
 }
 
static int _hoc_alpm() {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 
#if 1
 _check_alpm(_p, _ppvar, _thread, _nt);
#endif
 _r =  alpm ( _p, _ppvar, _thread, _nt, *getarg(1) ) ;
 ret(_r);
}
 static double _mfac_betm, _tmin_betm;
  static void _check_betm(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  if (!usetable) {return;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_betm =  - 150.0 ;
   _tmax =  150.0 ;
   _dx = (_tmax - _tmin_betm)/200.; _mfac_betm = 1./_dx;
   for (_i=0, _x=_tmin_betm; _i < 201; _x += _dx, _i++) {
    _t_betm[_i] = _f_betm(_p, _ppvar, _thread, _nt, _x);
   }
  }
 }

 double betm(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _lv) { 
#if 0
_check_betm(_p, _ppvar, _thread, _nt);
#endif
 return _n_betm(_p, _ppvar, _thread, _nt, _lv);
 }

 static double _n_betm(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 return _f_betm(_p, _ppvar, _thread, _nt, _lv); 
}
 _xi = _mfac_betm * (_lv - _tmin_betm);
 _i = (int) _xi;
 if (_xi <= 0.) {
 return _t_betm[0];
 }
 if (_i >= 200) {
 return _t_betm[200];
 }
 return _t_betm[_i] + (_xi - (double)_i)*(_t_betm[_i+1] - _t_betm[_i]);
 }

 
double _f_betm ( _p, _ppvar, _thread, _nt, _lv ) double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt; 
	double _lv ;
 {
   double _lbetm;
 _lbetm = 0.046 * exp ( - _lv / 22.73 ) ;
   
return _lbetm;
 }
 
static int _hoc_betm() {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 
#if 1
 _check_betm(_p, _ppvar, _thread, _nt);
#endif
 _r =  betm ( _p, _ppvar, _thread, _nt, *getarg(1) ) ;
 ret(_r);
}
 
static int  rates ( _p, _ppvar, _thread, _nt, _lv ) double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt; 
	double _lv ;
 {
   double _la ;
 _la = alpm ( _threadargscomma_ _lv ) ;
   taum = 1.0 / ( tfa * ( _la + betm ( _threadargscomma_ _lv ) ) ) ;
   minf = _la / ( _la + betm ( _threadargscomma_ _lv ) ) ;
   _la = alph ( _threadargscomma_ _lv ) ;
   tauh = 1.0 / ( tfi * ( _la + beth ( _threadargscomma_ _lv ) ) ) ;
   hinf = _la / ( _la + beth ( _threadargscomma_ _lv ) ) ;
    return 0; }
 
static int _hoc_rates() {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 rates ( _p, _ppvar, _thread, _nt, *getarg(1) ) ;
 ret(_r);
}
 
static int _ode_count(_type) int _type;{ return 2;}
 
static int _ode_spec(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  cai = _ion_cai;
  cao = _ion_cao;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
  }}
 
static int _ode_map(_ieq, _pv, _pvdot, _pp, _ppd, _atol, _type) int _ieq, _type; double** _pv, **_pvdot, *_pp, *_atol; Datum* _ppd; { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 2; ++_i) {
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
  cai = _ion_cai;
  cao = _ion_cao;
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 0, 1);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 1, 2);
   nrn_update_ion_pointer(_Ca_sym, _ppvar, 2, 3);
   nrn_update_ion_pointer(_Ca_sym, _ppvar, 3, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
  int _i; double _save;{
  h = h0;
  m = m0;
 {
   rates ( _threadargscomma_ v ) ;
   m = minf ;
   h = hinf ;
   gcat = gcatbar * m * m * h * h2 ( _threadargscomma_ cai ) ;
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
 _check_alph(_p, _ppvar, _thread, _nt);
 _check_beth(_p, _ppvar, _thread, _nt);
 _check_alpm(_p, _ppvar, _thread, _nt);
 _check_betm(_p, _ppvar, _thread, _nt);
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
  cai = _ion_cai;
  cao = _ion_cao;
 initmodel(_p, _ppvar, _thread, _nt);
 }}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   gcat = gcatbar * m * m * h * h2 ( _threadargscomma_ cai ) ;
   iCa = gcat * ghk ( _threadargscomma_ v , cai , cao ) ;
   }
 _current += iCa;

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
  cai = _ion_cai;
  cao = _ion_cao;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _diCa;
  _diCa = iCa;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_diCadv += (_diCa - iCa)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_iCa += iCa ;
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
  cai = _ion_cai;
  cao = _ion_cao;
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
 _slist1[0] = &(m) - _p;  _dlist1[0] = &(Dm) - _p;
 _slist1[1] = &(h) - _p;  _dlist1[1] = &(Dh) - _p;
   _t_alph = makevector(201*sizeof(double));
   _t_beth = makevector(201*sizeof(double));
   _t_alpm = makevector(201*sizeof(double));
   _t_betm = makevector(201*sizeof(double));
_first = 0;
}
