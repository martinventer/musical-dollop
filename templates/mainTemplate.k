$# Created on $date_time
*KEYWORD
*PARAMETER
$#   prmr1      val1     prmr2      val2     prmr3      val3     prmr4      val4
Rmod      200e9                                                                 
*TITLE_TITLE
$#                                                                         title
$title
*CONTROL_IMPLICIT_AUTO
$#   iauto    iteopt    itewin     dtmin     dtmax     dtexp     kfail    kcycle
         0        11         5       0.0       0.1       0.0         0         0
*CONTROL_IMPLICIT_GENERAL
$#  imflag       dt0    imform      nsbs       igs     cnstn      form    zero_v
         1      0.05         2         0         1         0         0         0
*CONTROL_IMPLICIT_SOLUTION
$#  nsolvr    ilimit    maxref     dctol     ectol     rctol     lstol    abstol
        12        11        20     0.001      0.011.00000E10       0.91.0000E-10
$#   dnorm    diverg     istif   nlprint    nlnorm   d3itctl     cpchk     
         2         1         1         0         2         0         0
$#  arcctl    arcdir    arclen    arcmth    arcdmp    arcpsi    arcalf    arctim
         0         0       0.0         1         2         0         0         0
$#   lsmtd     lsdir      irad      srad      awgt      sred    
         4         2       0.0       0.0       0.0       0.0
*CONTROL_IMPLICIT_SOLVER
$#  lsolvr    lprint     negev     order      drcm    drcprm   autospc   autotol
         2         0         2         0         4       0.0         1       0.0
$#  lcpack    mtxdmp    iparm1    rparm1    rparm2      
         2         0       5001.00000E-9     0.001
$#  emxdmp    rdcmem      
         0      0.85
*CONTROL_TERMINATION
$#  endtim    endcyc     dtmin    endeng    endmas     nosol     
       1.0         0       0.0       0.01.000000E8         0
*DATABASE_BNDOUT
$#      dt    binary      lcur     ioopt     
       0.1         3         0         1
*DATABASE_GLSTAT
$#      dt    binary      lcur     ioopt     
       0.1         3         0         1
*DATABASE_MATSUM
$#      dt    binary      lcur     ioopt     
       0.1         3         0         1
*DATABASE_NODFOR
$#      dt    binary      lcur     ioopt     
       0.1         3         0         1
*DATABASE_NODOUT
$#      dt    binary      lcur     ioopt   option1   option2       
       0.1         3         0         1       0.0         0
*DATABASE_SPCFORC
$#      dt    binary      lcur     ioopt     
       0.1         3         0         1
*DATABASE_BINARY_D3PLOT
$#      dt      lcdt      beam     npltc    psetid      
      0.05         0         0         0         0
$#   ioopt     
         0
*DATABASE_EXTENT_BINARY
$#   neiph     neips    maxint    strflg    sigflg    epsflg    rltflg    engflg
         0         0         3         1         1         1         1         1
$#  cmpflg    ieverp    beamip     dcomp      shge     stssz    n3thdt   ialemat
         0         0         0         1         1         1         2         1
$# nintsld   pkp_sen      sclp     hydro     msscl     therm    intout    nodout
         0         0       1.0         0         0         0                    
$#    dtdt    resplt     neipb     quadr     cubic     
         0         0         0         0         0
*DATABASE_FORMAT
$#   iform   ibinary       
         0         1
*SECTION_SOLID_TITLE
puck elements
$#   secid    elform       aet   
         1         1         0
*MAT_ELASTIC_TITLE
linMat
$#     mid        ro         e        pr        da        db  not used        
         1    7980.02.00000E11       0.3       0.0       0.0       0.0
*DEFINE_CURVE_TITLE
MovingBC
$#    lcid      sidr       sfa       sfo      offa      offo    dattyp     lcint
         1         0       1.0     0.002       0.0       0.0         0         0
$#                a1                  o1  
                 0.0                 0.0
                 1.0                 1.0
                 2.0                 2.0
*BOUNDARY_PRESCRIBED_MOTION_SET
         2         1         2         1      -1.0         01.00000E28       0.0
*BOUNDARY_SPC_SET
         1         0         1         1         1         1         1         1
*BOUNDARY_SPC_SET
         2         0         0         1         1         0         0         0
*INCLUDE
$include_file
*END
