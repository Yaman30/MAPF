Att följa ideala trajektorier med kinodynamiskt begränsade robotar

Detta projekt innehåller koden som användes i vårt kandidatarbete vid Chalmers tekniska högskola.
Syftet med projektet är att utveckla ett simuleringsverktyg i Python för att visualisera och analysera rörelseplanering för flera agenter.

Nedan beskrivs de viktigaste filerna och mapparna som används i projektet:
- LQR.py : kör simulering med LQR.
- LQR_CBF.py : kör simulering med LQR och CBF.
- MPC.py : kör simulering med MPC.
- MPC_CBF.py : kör simulering med MPC och CBF.
- ReferenceTrajectoryUtils.py : läser in JSON-filer och skapar tidsberoende referensbanor för robotarna.
- Instances : innehåller kartor och lösningar för testfallen.

Koden är skriven i Python. Följande bibliotek som behövs:
- import matplotlib.pyplot as plt
- import matplotlib.patches as patches
- import numpy as np
- from matplotlib.animation import FuncAnimation
- from scipy.linalg import solve_discrete_are
- import casadi as ca
- from ReferenceTrajectoryUtils import get_instance_trajectories

Ahmed Yaman Alkhuzaee  
Eyad Tahhan  

Chalmers tekniska högskola, 2026
