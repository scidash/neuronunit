#include <stdio.h>
#include "hocdec.h"
#define IMPORT extern __declspec(dllimport)
IMPORT int nrnmpi_myid, nrn_nobanner_;

modl_reg(){
	//nrn_mswindll_stdio(stdin, stdout, stderr);
    if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
	fprintf(stderr, "Additional mechanisms from files\n");

fprintf(stderr," kadist.mod");
fprintf(stderr," kaprox.mod");
fprintf(stderr," kdrca1.mod");
fprintf(stderr," nax.mod");
fprintf(stderr," nmda.mod");
fprintf(stderr," vmax.mod");
fprintf(stderr, "\n");
    }
_kadist_reg();
_kaprox_reg();
_kdrca1_reg();
_nax_reg();
_nmda_reg();
_vmax_reg();
}
