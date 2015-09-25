#include <stdio.h>
#include "hocdec.h"
#define IMPORT extern __declspec(dllimport)
IMPORT int nrnmpi_myid, nrn_nobanner_;

modl_reg(){
	//nrn_mswindll_stdio(stdin, stdout, stderr);
    if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
	fprintf(stderr, "Additional mechanisms from files\n");

fprintf(stderr," cad.mod");
fprintf(stderr," cagk.mod");
fprintf(stderr," cal.mod");
fprintf(stderr," calH.mod");
fprintf(stderr," car.mod");
fprintf(stderr," cat.mod");
fprintf(stderr," d3.mod");
fprintf(stderr," h.mod");
fprintf(stderr," kadist.mod");
fprintf(stderr," kaprox.mod");
fprintf(stderr," kca.mod");
fprintf(stderr," kdr.mod");
fprintf(stderr," km.mod");
fprintf(stderr," na3.mod");
fprintf(stderr," na3dend.mod");
fprintf(stderr," na3notrunk.mod");
fprintf(stderr," nap.mod");
fprintf(stderr," nax.mod");
fprintf(stderr," nmda.mod");
fprintf(stderr," somacar.mod");
fprintf(stderr," vmax.mod");
fprintf(stderr, "\n");
    }
_cad_reg();
_cagk_reg();
_cal_reg();
_calH_reg();
_car_reg();
_cat_reg();
_d3_reg();
_h_reg();
_kadist_reg();
_kaprox_reg();
_kca_reg();
_kdr_reg();
_km_reg();
_na3_reg();
_na3dend_reg();
_na3notrunk_reg();
_nap_reg();
_nax_reg();
_nmda_reg();
_somacar_reg();
_vmax_reg();
}
