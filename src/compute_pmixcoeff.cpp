// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "compute_coord_atom.h"

#include "atom.h"
#include "comm.h"
#include "compute_orientorder_atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "universe.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <iostream>

#include "plumed/wrapper/Plumed.h"

#if defined(__PLUMED_DEFAULT_KERNEL)
#define PLUMED_QUOTE_DIRECT(name) #name
#define PLUMED_QUOTE(macro) PLUMED_QUOTE_DIRECT(macro)
static const char plumed_default_kernel[] = "PLUMED_KERNEL=" PLUMED_QUOTE(__PLUMED_DEFAULT_KERNEL);
#endif

/* --------------------------------------------------------------- */

using namespace LAMMPS_NS;


/* ---------------------------------------------------------------------- */

ComputeCoordAtom::ComputeCoordAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  typelo(nullptr), typehi(nullptr), cvec(nullptr), carray(nullptr),
  group2(nullptr), id_orientorder(nullptr), normv(nullptr)
{
  if (narg < 5) error->all(FLERR,"Illegal compute coord/atom command");

  jgroup = group->find("all");
  jgroupbit = group->bitmask[jgroup];
  cstyle = NONE;

  if (strcmp(arg[3],"cutoff") == 0) {
    cstyle = CUTOFF;
    double cutoff = utils::numeric(FLERR,arg[4],false,lmp);
    cutsq = cutoff*cutoff;

    int iarg = 5;
    if ((narg > 6) && (strcmp(arg[5],"group") == 0)) {
      group2 = utils::strdup(arg[6]);
      iarg += 2;
      jgroup = group->find(group2);
      if (jgroup == -1)
        error->all(FLERR,"Compute coord/atom group2 ID does not exist");
      jgroupbit = group->bitmask[jgroup];
    }

    ncol = narg-iarg + 1;
    int ntypes = atom->ntypes;
    typelo = new int[ncol];
    typehi = new int[ncol];

    if (narg == iarg) {
      ncol = 1;
      typelo[0] = 1;
      typehi[0] = ntypes;
    } else {
      ncol = 0;
      while (iarg < narg) {
        utils::bounds(FLERR,arg[iarg],1,ntypes,typelo[ncol],typehi[ncol],error);
        if (typelo[ncol] > typehi[ncol])
          error->all(FLERR,"Illegal compute coord/atom command");
        ncol++;
        iarg++;
      }
    }

  } else if (strcmp(arg[3],"orientorder") == 0) {
    cstyle = ORIENT;
    if (narg != 6) error->all(FLERR,"Illegal compute coord/atom command");

    id_orientorder = utils::strdup(arg[4]);

    int iorientorder = modify->find_compute(id_orientorder);
    if (iorientorder < 0)
      error->all(FLERR,"Could not find compute coord/atom compute ID");
    if (!utils::strmatch(modify->compute[iorientorder]->style,"^orientorder/atom"))
      error->all(FLERR,"Compute coord/atom compute ID is not orientorder/atom");

    threshold = utils::numeric(FLERR,arg[5],false,lmp);
    if (threshold <= -1.0 || threshold >= 1.0)
      error->all(FLERR,"Compute coord/atom threshold not between -1 and 1");

    ncol = 1;
    typelo = new int[ncol];
    typehi = new int[ncol];
    typelo[0] = 1;
    typehi[0] = atom->ntypes;

  } else if (strcmp(arg[3],"plumed") == 0) {
    cstyle = PLUMED;
    std::cout<<arg[3]<<std::endl;
    if (narg != 8) error->all(FLERR,"Illegal compute coord/atom command");

    // id_orientorder = utils::strdup(arg[4]);
    // 
    // int iorientorder = modify->find_compute(id_orientorder);
    // if (iorientorder < 0)
    //   error->all(FLERR,"Could not find compute coord/atom compute ID");
    // if (!utils::strmatch(modify->compute[iorientorder]->style,"^orientorder/atom"))
    //   error->all(FLERR,"Compute coord/atom compute ID is not orientorder/atom");
    if (!atom->tag_enable)
      error->all(FLERR,"Fix plumed requires atom tags");

    if (atom->tag_consecutive() == 0)
      error->all(FLERR,"Fix plumed requires consecutive atom IDs");

    if (igroup != 0 && comm->me == 0)
      error->warning(FLERR,"Fix group for fix plumed is not 'all'. "
                   "Group will be ignored.");

#if defined(__PLUMED_DEFAULT_KERNEL)
    if (getenv("PLUMED_KERNEL") == nullptr)
      platform::putenv(plumed_default_kernel);
#endif

    c_plumed=new PLMD::Plumed;
    // Check API version

    int api_version=0;
    c_plumed->cmd("getApiVersion",&api_version);
    if ((api_version < 5) || (api_version > 8))
      error->all(FLERR,"Incompatible API version for PLUMED in fix plumed. "
                 "Only Plumed 2.4.x, 2.5.x, 2.6.x, 2.7.x are tested and supported.");

#if !defined(MPI_STUBS)
    // If the -partition option is activated then enable
    // inter-partition communication

    if (universe->existflag == 1) {
      int me;
      MPI_Comm inter_comm;
      MPI_Comm_rank(world,&me);

    // Change MPI_COMM_WORLD to universe->uworld which seems more appropriate

      MPI_Comm_split(universe->uworld,me,0,&inter_comm);
      c_plumed->cmd("GREX setMPIIntracomm",&world);
      if (me == 0) {
        // The inter-partition communicator is only defined for the root in
        //    each partition (a.k.a. world). This is due to the way in which
        //    it is defined inside plumed.
        c_plumed->cmd("GREX setMPIIntercomm",&inter_comm);
      }
      c_plumed->cmd("GREX init",nullptr);
    }

    // The general communicator is independent of the existence of partitions,
    // if there are partitions, world is defined within each partition,
    // whereas if partitions are not defined then world is equal to
    // MPI_COMM_WORLD.

    // plumed does not know about LAMMPS using the MPI STUBS library and will
    // fail if this is called under these circumstances
    c_plumed->cmd("setMPIComm",&world);
#endif
    // Set up units
    // LAMMPS units wrt kj/mol - nm - ps
    // Set up units

    if (strcmp(update->unit_style,"lj") == 0) {
      // LAMMPS units lj
      c_plumed->cmd("setNaturalUnits");
    } else {

      // Conversion factor from LAMMPS energy units to kJ/mol (units of PLUMED)

      double energyUnits=1.0;

      // LAMMPS units real :: kcal/mol;

      if (strcmp(update->unit_style,"real") == 0) {
        energyUnits=4.184;

        // LAMMPS units metal :: eV;
 
      } else if (strcmp(update->unit_style,"metal") == 0) {
        energyUnits=96.48530749925792;

        // LAMMPS units si :: Joule;

      } else if (strcmp(update->unit_style,"si") == 0) {
        energyUnits=0.001;

        // LAMMPS units cgs :: erg;

      } else if (strcmp(update->unit_style,"cgs") == 0) {
        energyUnits=6.0221418e13;

        // LAMMPS units electron :: Hartree;

      } else if (strcmp(update->unit_style,"electron") == 0) {
        energyUnits=2625.5257;

      } else error->all(FLERR,"Fix plumed cannot handle your choice of units");

      // Conversion factor from LAMMPS length units to nm (units of PLUMED)

      double lengthUnits=0.1/force->angstrom;

      // Conversion factor from LAMMPS time unit to ps (units of PLUMED)

      double timeUnits=0.001/force->femtosecond;

      c_plumed->cmd("setMDEnergyUnits",&energyUnits);
      c_plumed->cmd("setMDLengthUnits",&lengthUnits);
      c_plumed->cmd("setMDTimeUnits",&timeUnits);
    }

    // Read compute parameters:

    int next=0;
    for (int i=4;i<narg;++i) {
      std::cout<<arg[i]<<std::endl;
      std::cout<<narg<<std::endl;
      if (!strcmp(arg[i],"outfile")) {
        next=1;
      } else if (next==1) {
        if (universe->existflag == 1) {
          // Each replica writes an independent log file
          //  with suffix equal to the replica id
          c_plumed->cmd("setLogFile",fmt::format("{}.{}",arg[i],universe->iworld).c_str());
          next=0;
        } else {
          // partition option not used
          c_plumed->cmd("setLogFile",arg[i]);
          next=0;
        }
      } else if (!strcmp(arg[i],"plumedfile")) {
        next=2;
      } else if (next==2) {
        c_plumed->cmd("setPlumedDat",arg[i]);
        next=0;
      } else error->all(FLERR,"Syntax error - use 'fix <fix-ID> plumed "
                       "plumedfile plumed.dat outfile plumed.out' ");
    }
    if (next==1) error->all(FLERR,"missing argument for outfile option");
    if (next==2) error->all(FLERR,"missing argument for plumedfile option");

    c_plumed->cmd("setMDEngine","LAMMPS");
 
    if (atom->natoms > MAXSMALLINT)
        error->all(FLERR,"Fix plumed can only handle up to 2.1 billion atoms");

    natoms=int(atom->natoms);
    c_plumed->cmd("setNatoms",&natoms);

    double dt=update->dt;
    c_plumed->cmd("setTimestep",&dt);

    // threshold = utils::numeric(FLERR,arg[5],false,lmp);
    // if (threshold <= -1.0 || threshold >= 1.0)
    //   error->all(FLERR,"Compute coord/atom threshold not between -1 and 1");

    ncol = 1;
    typelo = new int[ncol];
    typehi = new int[ncol];
    typelo[0] = 1;
    typehi[0] = atom->ntypes;

  } else error->all(FLERR,"Invalid cstyle in compute coord/atom");

  peratom_flag = 1;
  if (ncol == 1) size_peratom_cols = 0;
  else size_peratom_cols = ncol;

  nmax = 0;
}

/* ---------------------------------------------------------------------- */

ComputeCoordAtom::~ComputeCoordAtom()
{
  if (copymode) return;

  delete [] group2;
  delete [] typelo;
  delete [] typehi;
  memory->destroy(cvec);
  memory->destroy(carray);
  delete [] id_orientorder;
  delete c_plumed;
  // delete [] id_pe;
  // delete [] id_press;
}

/* ---------------------------------------------------------------------- */

void ComputeCoordAtom::init()
{
  if (cstyle == ORIENT) {
    int iorientorder = modify->find_compute(id_orientorder);
    c_orientorder = (ComputeOrientOrderAtom*)(modify->compute[iorientorder]);
    cutsq = c_orientorder->cutsq;
    l = c_orientorder->qlcomp;
    //  communicate real and imaginary 2*l+1 components of the normalized vector
    comm_forward = 2*(2*l+1);
    if (!(c_orientorder->qlcompflag))
      error->all(FLERR,"Compute coord/atom requires components "
                 "option in compute orientorder/atom");
  }
  if (cstyle == PLUMED){
    c_plumed->cmd("init");
  }

  if (force->pair == nullptr)
    error->all(FLERR,"Compute coord/atom requires a pair style be defined");
  if (sqrt(cutsq) > force->pair->cutforce)
    error->all(FLERR,
               "Compute coord/atom cutoff is longer than pairwise cutoff");

  // need an occasional full neighbor list

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 1;
}

/* ---------------------------------------------------------------------- */

void ComputeCoordAtom::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeCoordAtom::compute_peratom()
{
  int i,j,m,ii,jj,inum,jnum,jtype,n;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double *count;

  invoked_peratom = update->ntimestep;

  // grow coordination array if necessary

  if (atom->nmax > nmax) {
    if (ncol == 1) {
      memory->destroy(cvec);
      nmax = atom->nmax;
      memory->create(cvec,nmax,"coord/atom:cvec");
      vector_atom = cvec;
    } else {
      memory->destroy(carray);
      nmax = atom->nmax;
      memory->create(carray,nmax,ncol,"coord/atom:carray");
      array_atom = carray;
    }
  }

  if (cstyle == ORIENT) {
    if (!(c_orientorder->invoked_flag & Compute::INVOKED_PERATOM)) {
      c_orientorder->compute_peratom();
      c_orientorder->invoked_flag |= Compute::INVOKED_PERATOM;
    }
    nqlist = c_orientorder->nqlist;
    normv = c_orientorder->array_atom;
    comm->forward_comm_compute(this);
  }

  if (cstyle == PLUMED){

    int update_gatindex=0;
   
    if (nlocal != atom->nlocal) {
   
      if (charges) delete [] charges;
      if (masses) delete [] masses;
      if (gatindex) delete [] gatindex;
   
      nlocal=atom->nlocal;
      gatindex=new int [nlocal];
      masses=new double [nlocal];
      charges=new double [nlocal];
      update_gatindex=1;
   
    } else {
   
      for (int i=0;i<nlocal;i++) {
        if (gatindex[i]!=atom->tag[i]-1) {
          update_gatindex=1;
          break;
        }
      }
    }
    MPI_Allreduce(MPI_IN_PLACE,&update_gatindex,1,MPI_INT,MPI_SUM,world);
   
    // In case it has been updated, rebuild the local mass/charges array
    // and tell plumed about the change:
   
    if (update_gatindex) {
      for (int i=0;i<nlocal;i++) gatindex[i]=atom->tag[i]-1;
      // Get masses
      if (atom->rmass_flag) {
        for (int i=0;i<nlocal;i++) masses[i]=atom->rmass[i];
      } else {
        for (int i=0;i<nlocal;i++) masses[i]=atom->mass[atom->type[i]];
      }
      // Get charges
      if (atom->q_flag) {
        for (int i=0;i<nlocal;i++) charges[i]=atom->q[i];
      } else {
        for (int i=0;i<nlocal;i++) charges[i]=0.0;
      }
      c_plumed->cmd("setAtomsNlocal",&nlocal);
      c_plumed->cmd("setAtomsGatindex",gatindex);
    }

    double box[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) box[i][j]=0.0;
    box[0][0]=domain->h[0];
    box[1][1]=domain->h[1];
    box[2][2]=domain->h[2];
    box[2][1]=domain->h[3];
    box[2][0]=domain->h[4];
    box[1][0]=domain->h[5];

    int step=int(update->ntimestep);
    // pass all pointers to plumed:
    c_plumed->cmd("setStep",&step);
    int plumedStopCondition=0;
    c_plumed->cmd("setStopFlag",&plumedStopCondition);
    c_plumed->cmd("setPositions",&atom->x[0][0]);
    c_plumed->cmd("setBox",&box[0][0]);
    // c_plumed->cmd("setForces",&atom->f[0][0]);
    c_plumed->cmd("setMasses",&masses[0]);
    c_plumed->cmd("setCharges",&charges[0]);
    // c_plumed->cmd("setVirial",&plmd_virial[0][0]);
    c_plumed->cmd("prepareCalc");

    plumedNeedsEnergy=0;
    c_plumed->cmd("isEnergyNeeded",&plumedNeedsEnergy);
  }

  // invoke full neighbor list (will copy or build if necessary)

  neighbor->build_one(list);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // compute coordination number(s) for each atom in group
  // use full neighbor list to count atoms less than cutoff

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;

  if (cstyle == CUTOFF) {

    if (ncol == 1) {

      for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        if (mask[i] & groupbit) {
          xtmp = x[i][0];
          ytmp = x[i][1];
          ztmp = x[i][2];
          jlist = firstneigh[i];
          jnum = numneigh[i];

          n = 0;
          for (jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            j &= NEIGHMASK;

            if (mask[j] & jgroupbit) {
              jtype = type[j];
              delx = xtmp - x[j][0];
              dely = ytmp - x[j][1];
              delz = ztmp - x[j][2];
              rsq = delx*delx + dely*dely + delz*delz;
              if (rsq < cutsq && jtype >= typelo[0] && jtype <= typehi[0])
                n++;
            }
          }

          cvec[i] = n;
        } else cvec[i] = 0.0;
      }

    } else {
      for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        count = carray[i];
        for (m = 0; m < ncol; m++) count[m] = 0.0;

        if (mask[i] & groupbit) {
          xtmp = x[i][0];
          ytmp = x[i][1];
          ztmp = x[i][2];
          jlist = firstneigh[i];
          jnum = numneigh[i];


          for (jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            j &= NEIGHMASK;

            jtype = type[j];
            delx = xtmp - x[j][0];
            dely = ytmp - x[j][1];
            delz = ztmp - x[j][2];
            rsq = delx*delx + dely*dely + delz*delz;
            if (rsq < cutsq) {
              for (m = 0; m < ncol; m++)
                if (jtype >= typelo[m] && jtype <= typehi[m])
                  count[m] += 1.0;
            }
          }
        }
      }
    }

  } else if (cstyle == ORIENT) {

    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      if (mask[i] & groupbit) {
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];
        jlist = firstneigh[i];
        jnum = numneigh[i];

        n = 0;
        for (jj = 0; jj < jnum; jj++) {
          j = jlist[jj];
          j &= NEIGHMASK;
          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = ztmp - x[j][2];
          rsq = delx*delx + dely*dely + delz*delz;
          if (rsq < cutsq) {
            double dot_product = 0.0;
            for (m=0; m < 2*(2*l+1); m++) {
              dot_product += normv[i][nqlist+m]*normv[j][nqlist+m];
            }
            if (dot_product > threshold) n++;
          }
        }
        cvec[i] = n;
      } else cvec[i] = 0.0;
    }
  } else if (cstyle == PLUMED) {
    c_plumed->cmd("performCalcNoUpdate");
    unsigned nquantities[1]={};
    c_plumed->cmd("getNumberOfQuantities", nquantities);
    unsigned nderivatives[1]={};
    c_plumed->cmd("getNumberOfDerivatives", nderivatives);
    // assert(*nquantities == 28);
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      const double* MCV=NULL;
      c_plumed->cmd(("getMultiColvars "+std::to_string(i)).c_str(), &MCV);
      std::cout<<"MCV of Atom "<<std::to_string(i)<<std::endl;
      for(int i=0; i<*nquantities; ++i){
        std::cout<<MCV[i]<<"  ";
      }
      std::cout<<std::endl;
    }
  }
}

/* ---------------------------------------------------------------------- */

int ComputeCoordAtom::pack_forward_comm(int n, int *list, double *buf,
                                        int /*pbc_flag*/, int * /*pbc*/)
{
  int i,m=0,j;
  for (i = 0; i < n; ++i) {
    for (j = nqlist; j < nqlist + 2*(2*l+1); ++j) {
      buf[m++] = normv[list[i]][j];
    }
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeCoordAtom::unpack_forward_comm(int n, int first, double *buf)
{
  int i,last,m=0,j;
  last = first + n;
  for (i = first; i < last; ++i) {
    for (j = nqlist; j < nqlist + 2*(2*l+1); ++j) {
      normv[i][j] = buf[m++];
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeCoordAtom::memory_usage()
{
  double bytes = (double)ncol*nmax * sizeof(double);
  return bytes;
}
