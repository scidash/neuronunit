#include <stdio.h>
#include "hocdec.h"
#define IMPORT extern __declspec(dllimport)
IMPORT int nrnmpi_myid, nrn_nobanner_;

modl_reg(){
	//nrn_mswindll_stdio(stdin, stdout, stderr);
    if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
	fprintf(stderr, "Additional mechanisms from files\n");

fprintf(stderr," CaL.mod");
fprintf(stderr," CaN.mod");
fprintf(stderr," CaR.mod");
fprintf(stderr," CaT.mod");
fprintf(stderr," H_CA1pyr_dist.mod");
fprintf(stderr," H_CA1pyr_prox.mod");
fprintf(stderr," K_AHP3.mod");
fprintf(stderr," K_A_dist.mod");
fprintf(stderr," K_A_prox.mod");
fprintf(stderr," K_C_1D.mod");
fprintf(stderr," K_DRS4.mod");
fprintf(stderr," K_M.mod");
fprintf(stderr," K_M2.mod");
fprintf(stderr," Leak_pyr.mod");
fprintf(stderr," NMDA_JS.mod");
fprintf(stderr," NMDA_M.mod");
fprintf(stderr," Na_dend13.mod");
fprintf(stderr," Na_soma13.mod");
fprintf(stderr," cacum.mod");
fprintf(stderr," nmda.mod");
fprintf(stderr," nmda_dsyn.mod");
fprintf(stderr," nmda_dsyn_2.mod");
fprintf(stderr," vecevent.mod");
fprintf(stderr," vmax.mod");
fprintf(stderr, "\n");
    }
_CaL_reg();
_CaN_reg();
_CaR_reg();
_CaT_reg();
_H_CA1pyr_dist_reg();
_H_CA1pyr_prox_reg();
_K_AHP3_reg();
_K_A_dist_reg();
_K_A_prox_reg();
_K_C_1D_reg();
_K_DRS4_reg();
_K_M_reg();
_K_M2_reg();
_Leak_pyr_reg();
_NMDA_JS_reg();
_NMDA_M_reg();
_Na_dend13_reg();
_Na_soma13_reg();
_cacum_reg();
_nmda_reg();
_nmda_dsyn_reg();
_nmda_dsyn_2_reg();
_vecevent_reg();
_vmax_reg();
}
