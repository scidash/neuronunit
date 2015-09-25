/* Created by Language version: 6.2.0 */
/* NOT VECTORIZED */
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
 
#define _threadargscomma_ /**/
#define _threadargs_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define gmax _p[0]
#define i _p[1]
#define factor _p[2]
#define t_last _p[3]
#define A _p[4]
#define B _p[5]
#define mgB _p[6]
#define tp _p[7]
#define DA _p[8]
#define DB _p[9]
#define DmgB _p[10]
#define _g _p[11]
#define _tsav _p[12]
#define _nd_area  *_ppvar[0]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 static int hoc_nrnpointerindex =  -1;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_mgblock();
 static int _mechtype;
extern int nrn_get_mechtype();
 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(_ho) Object* _ho; { void* create_point_process();
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt();
 static double _hoc_loc_pnt(_vptr) void* _vptr; {double loc_point_process();
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(_vptr) void* _vptr; {double has_loc_point();
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(_vptr)void* _vptr; {
 double get_loc_point_process(); return (get_loc_point_process(_vptr));
}
 static _hoc_setdata(_vptr) void* _vptr; { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 /* connect user functions to hoc names */
 static IntFunc hoc_intfunc[] = {
 0,0
};
 static struct Member_func {
	char* _name; double (*_member)();} _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 "mgblock", _hoc_mgblock,
 0, 0
};
 /* declare global and static user variables */
#define Deadtime Deadtime_NMDA_dsyn_2
 double Deadtime = 0;
#define Prethresh Prethresh_NMDA_dsyn_2
 double Prethresh = 0;
#define g g_NMDA_dsyn_2
 double g = 0;
#define mgB mgB_NMDA_dsyn_2
 double mgB = 0;
#define tau2 tau2_NMDA_dsyn_2
 double tau2 = 2;
#define tau1 tau1_NMDA_dsyn_2
 double tau1 = 75;
#define usetable usetable_NMDA_dsyn_2
 double usetable = 1;
#define vmax vmax_NMDA_dsyn_2
 double vmax = 100;
#define vmin vmin_NMDA_dsyn_2
 double vmin = -120;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "usetable_NMDA_dsyn_2", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "tau1_NMDA_dsyn_2", "ms",
 "tau2_NMDA_dsyn_2", "ms",
 "Prethresh_NMDA_dsyn_2", "mV",
 "Deadtime_NMDA_dsyn_2", "ms",
 "vmin_NMDA_dsyn_2", "mV",
 "vmax_NMDA_dsyn_2", "mV",
 "g_NMDA_dsyn_2", "nS",
 "gmax", "nS",
 "i", "nA",
 "t_last", "ms",
 0,0
};
 static double A0 = 0;
 static double B0 = 0;
 static double delta_t = 0.01;
 static double mgB0 = 0;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "tau1_NMDA_dsyn_2", &tau1_NMDA_dsyn_2,
 "tau2_NMDA_dsyn_2", &tau2_NMDA_dsyn_2,
 "Prethresh_NMDA_dsyn_2", &Prethresh_NMDA_dsyn_2,
 "Deadtime_NMDA_dsyn_2", &Deadtime_NMDA_dsyn_2,
 "vmin_NMDA_dsyn_2", &vmin_NMDA_dsyn_2,
 "vmax_NMDA_dsyn_2", &vmax_NMDA_dsyn_2,
 "g_NMDA_dsyn_2", &g_NMDA_dsyn_2,
 "mgB_NMDA_dsyn_2", &mgB_NMDA_dsyn_2,
 "usetable_NMDA_dsyn_2", &usetable_NMDA_dsyn_2,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(), nrn_init(), nrn_state();
 static void nrn_cur(), nrn_jacob();
 static void _hoc_destroy_pnt(_vptr) void* _vptr; {
   destroy_point_process(_vptr);
}
 
static int _ode_count(), _ode_map(), _ode_spec(), _ode_matsol();
 
#define _cvode_ieq _ppvar[2]._i
 /* connect range variables in _p that hoc is supposed to know about */
 static char *_mechanism[] = {
 "6.2.0",
"NMDA_dsyn_2",
 "gmax",
 0,
 "i",
 "factor",
 "t_last",
 0,
 "A",
 "B",
 "mgB",
 0,
 0};
 
static void nrn_alloc(_prop)
	Prop *_prop;
{
	Prop *prop_ion, *need_memb();
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 13, _prop);
 	/*initialize range parameters*/
 	gmax = 1.5;
  }
 	_prop->param = _p;
 	_prop->param_size = 13;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 3, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static _net_receive();
 typedef (*_Pfrv)();
 extern _Pfrv* pnt_receive;
 extern short* pnt_receive_size;
 _nmda_dsyn_2_reg() {
	int _vectorized = 0;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func,
	 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
  hoc_register_dparam_size(_mechtype, 3);
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 1;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 NMDA_dsyn_2 /cygdrive/d/SkyDrive/Dokumentumok/ITK MSc/CA1_piramissejt_dendrit/Modellek/Kali_Freund_modell/scppinput/nmda_dsyn_2.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double *_t_mgB;
static int _reset;
static char *modelname = "Diff-of-Exponential NMDA Synapse";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static _modl_cleanup(){ _match_recurse=1;}
static _f_mgblock();
static mgblock();
 
static int _ode_spec1(), _ode_matsol1();
 static _n_mgblock();
 static int _slist1[2], _dlist1[2];
 static int state();
 
/*CVODE*/
 static int _ode_spec1 () {_reset=0;
 {
   DA = - A / tau1 ;
   DB = - B / tau2 ;
   }
 return _reset;
}
 static int _ode_matsol1 () {
 DA = DA  / (1. - dt*( ( - 1.0 ) / tau1 )) ;
 DB = DB  / (1. - dt*( ( - 1.0 ) / tau2 )) ;
}
 /*END CVODE*/
 static int state () {_reset=0;
 {
    A = A + (1. - exp(dt*(( - 1.0 ) / tau1)))*(- ( 0.0 ) / ( ( - 1.0 ) / tau1 ) - A) ;
    B = B + (1. - exp(dt*(( - 1.0 ) / tau2)))*(- ( 0.0 ) / ( ( - 1.0 ) / tau2 ) - B) ;
   }
  return 0;
}
 static double _mfac_mgblock, _tmin_mgblock;
 static _check_mgblock();
 static _check_mgblock() {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  if (!usetable) {return;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_mgblock =  vmin ;
   _tmax =  vmax ;
   _dx = (_tmax - _tmin_mgblock)/400.; _mfac_mgblock = 1./_dx;
   for (_i=0, _x=_tmin_mgblock; _i < 401; _x += _dx, _i++) {
    _f_mgblock(_x);
    _t_mgB[_i] = mgB;
   }
  }
 }

 static mgblock(double _lv){ _check_mgblock();
 _n_mgblock(_lv);
 return;
 }

 static _n_mgblock(double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 _f_mgblock(_lv); return; 
}
 _xi = _mfac_mgblock * (_lv - _tmin_mgblock);
 _i = (int) _xi;
 if (_xi <= 0.) {
 mgB = _t_mgB[0];
 return; }
 if (_i >= 400) {
 mgB = _t_mgB[400];
 return; }
 _theta = _xi - (double)_i;
 mgB = _t_mgB[_i] + _theta*(_t_mgB[_i+1] - _t_mgB[_i]);
 }

 
static int  _f_mgblock (  _lv )  
	double _lv ;
 {
   mgB = 1.0 / ( 1.0 + 0.3 * exp ( - 0.1 * ( _lv ) ) ) ;
    return 0; }
 
static double _hoc_mgblock(_vptr) void* _vptr; {
 double _r;
    _hoc_setdata(_vptr);
  _r = 1.;
 mgblock (  *getarg(1) ) ;
 return(_r);
}
 
static _net_receive (_pnt, _args, _lflag) Point_process* _pnt; double* _args; double _lflag; 
{    _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t; {
   if ( t_last + Deadtime <= t ) {
     t_last = t ;
     A = A + _args[0] * factor ;
     B = B + _args[0] * factor ;
     }
   } }
 
static int _ode_count(_type) int _type;{ return 2;}
 
static int _ode_spec(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 ();
 }}
 
static int _ode_map(_ieq, _pv, _pvdot, _pp, _ppd, _atol, _type) int _ieq, _type; double** _pv, **_pvdot, *_pp, *_atol; Datum* _ppd; { 
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 2; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static int _ode_matsol(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
 _ode_matsol1 ();
 }}

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  A = A0;
  B = B0;
  mgB = mgB0;
 {
   i = 0.0 ;
   A = 0.0 ;
   B = 0.0 ;
   t_last = - 1.0 - Deadtime ;
   tp = ( tau2 * tau1 ) / ( tau1 - tau2 ) * log ( tau1 / tau2 ) ;
   factor = - exp ( - tp / tau2 ) + exp ( - tp / tau1 ) ;
   factor = 1.0 / factor ;
   mgblock ( _threadargscomma_ v ) ;
   }
  _sav_indep = t; t = _save;

}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _tsav = -1e20;
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
 initmodel();
}}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   mgblock ( _threadargscomma_ v ) ;
   g = gmax * mgB * ( A - B ) ;
   i = ( 1e-3 ) * g * v ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
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
 _g = _nrn_current(_v + .001);
 	{ _rhs = _nrn_current(_v);
 	}
 _g = (_g - _rhs)/.001;
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
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

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type){
 double _break, _save;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
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
 { {
 for (; t < _break; t += dt) {
 error =  state();
 if(error){fprintf(stderr,"at line 66 in file nmda_dsyn_2.mod:\n    SOLVE state METHOD cnexp\n"); nrn_complain(_p); abort_run(error);}
 
}}
 t = _save;
 }}}

}

static terminal(){}

static _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = &(A) - _p;  _dlist1[0] = &(DA) - _p;
 _slist1[1] = &(B) - _p;  _dlist1[1] = &(DB) - _p;
   _t_mgB = makevector(401*sizeof(double));
_first = 0;
}
