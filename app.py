import streamlit as st
from fractions import Fraction as F
import math
import itertools

def nums(s):
    v=[F(x) for x in s.replace(',',' ').split()]
    if not v: raise ValueError('Introduce números separados por espacios.')
    return v

def mat(s):
    a=[nums(l) for l in s.splitlines() if l.strip()]
    if not a or len(a)>12 or any(len(r)!=len(a) for r in a): raise ValueError('Matriz cuadrada de hasta 12 filas.')
    return a

def table(headers, rows):
    rows=[[str(x) for x in r] for r in rows]; h=list(headers)
    widths=[max(len(str(v)),*( [len(r[i]) for r in rows] or [0])) for i,v in enumerate(h)]
    line=lambda r:' | '.join(str(v).rjust(w) for v,w in zip(r,widths))
    return line(h)+'\n'+'-+-'.join('-'*w for w in widths)+'\n'+'\n'.join(line(r) for r in rows)

def showmat(m,n):
    return '\n'.join('[ '+' '.join(str(v).rjust(12) for v in r[:n])+' | '+' '.join(str(v).rjust(12) for v in r[n:])+' ]' for r in m)

def reduce(a,b,trace=False):
    n=len(a); m=[list(r)+list(t) for r,t in zip(a,b)]; log=[showmat(m,n)] if trace else []
    def record(s):
        if trace: log.extend([s,showmat(m,n)])
    for j in range(n):
        p=next((i for i in range(j,n) if m[i][j]),None)
        if p is None: raise ValueError('Matriz singular: no existe solución única ni inversa.')
        if p!=j: m[j],m[p]=m[p],m[j]; record(f'F{j+1} <-> F{p+1}')
        v=m[j][j]
        if v!=1: m[j]=[z/v for z in m[j]]; record(f'F{j+1} <- F{j+1} / ({v})')
        for i in range(n):
            if i!=j and m[i][j]:
                v=m[i][j]; m[i]=[z-v*w for z,w in zip(m[i],m[j])]; record(f'F{i+1} <- F{i+1} - ({v}) F{j+1}')
    return [r[n:] for r in m],log

def jordan(a,b,mode):
    n=len(a)
    if len(b)!=n: raise ValueError('Vector b incompatible.')
    inv=mode=='Inversa y solución'; rhs=[[F(i==j) for j in range(n)] for i in range(n)] if inv else [[v] for v in b]
    r,log=reduce(a,rhs,True); out=['Matriz [A | I] inicial' if inv else 'Matriz [A | b] inicial',*log]
    if inv:
        out+=['Matriz inversa A^-1',showmat(r,n).replace(' |  ]',' ]')]; x=[sum(u*v for u,v in zip(row,b)) for row in r]
        out += [f'x{i+1} = '+' + '.join(f'({u})({v})' for u,v in zip(row,b))+f' = {x[i]}' for i,row in enumerate(r)]
    else: x=[r0[0] for r0 in r]
    out+=['RESULTADO EXACTO']+[f'x{i+1} = {v} (aprox. {float(v):.8f})' for i,v in enumerate(x)]+['Comprobación Ax = b: '+('CORRECTA' if all(sum(u*v for u,v in zip(row,x))==z for row,z in zip(a,b)) else 'INCORRECTA')]
    return '\n\n'.join(out)

def seidel(a,b,start,tol,limit,omega,criterion,reorder):
    n=len(a)
    if len(b)!=n or len(start)!=n or not 0<tol or not 0<omega<2 or not 1<=limit<=2000: raise ValueError('Revisa dimensiones, tolerancia, omega (0 a 2) e iteraciones (1–2000).')
    direct,_=reduce(a,[[v] for v in b]); exact=[float(r[0]) for r in direct]
    a0,b0=a,b; order=list(range(n))
    if reorder=='Sí' and n<=7:
        def score(p): return sum(abs(a[j][i])>sum(abs(v) for k,v in enumerate(a[j]) if k!=i) for i,j in enumerate(p)) if all(a[j][i] for i,j in enumerate(p)) else -1
        order=list(max(itertools.permutations(range(n)),key=score))
    a=[[float(v) for v in a0[j]] for j in order]; b=[float(b0[j]) for j in order]; x=list(map(float,start))
    if any(not a[i][i] for i in range(n)): raise ValueError('Diagonal con ceros; intenta reordenar.')
    norm=lambda v:max(map(abs,v)); residual=lambda x:norm([sum(u*v for u,v in zip(r,x))-z for r,z in zip(a,b)])
    initial=residual(x); scale=norm(b); names=[f'x{i+1}' for i in range(n)]; f=lambda v:f'{v:.9f}'
    sys=lambda a,b:table(['Ecuación',*names,'b'],[[i+1,*map(str,r),str(z)] for i,(r,z) in enumerate(zip(a,b))])
    out=['SISTEMA ORIGINAL',sys(a0,b0),'Solución de comprobación: '+', '.join(f'{k}={v:.12f}' for k,v in zip(names,exact)), 'Nuevo orden: '+', '.join(str(j+1) for j in order),sys(a,b)]
    dominant=all(abs(r[i])>sum(abs(v) for j,v in enumerate(r) if j!=i) for i,r in enumerate(a)); out+=['Dominancia diagonal estricta: '+('SÍ' if dominant else 'NO; no se garantiza convergencia.'),'DESPEJES']
    out += [f'{names[i]} = ({b[i]} '+''.join(f' - ({a[i][j]})*{names[j]}' for j in range(n) if j!=i)+f') / ({a[i][i]})' for i in range(n)]
    out += [f'Omega={omega}; tolerancia={tol}; criterio={criterion}']; rows=[]; details=[]; stopped=False
    for k in range(1,limit+1):
        old=x.copy()
        for i in range(n):
            gs=(b[i]-sum(a[i][j]*x[j] for j in range(n) if i!=j))/a[i][i]; correction=omega*(gs-x[i]); x[i]+=correction; details.append([k,names[i],f(old[i]),f(gs),f(correction),f(x[i])])
        if not all(math.isfinite(v) for v in x): raise ValueError('Divergencia numérica.')
        err=norm([u-v for u,v in zip(x,old)]); rel=err/norm(x) if norm(x) else (0 if not err else math.inf); res=residual(x); rr=res/scale if scale else (0 if not res else math.inf)
        metrics=[err,rel,100*rel,res,rr]; rows.append([k,*map(f,x),*map(f,metrics)])
        if metrics[['Error absoluto','Error relativo','Error %','Residuo absoluto','Residuo relativo'].index(criterion)]<=tol: stopped=True; break
    out+=['TABLA DE ITERACIONES',table(['Iter.',*names,'Error abs.','Error rel.','Error %','Residuo abs.','Residuo rel.'],rows),'TABLA DETALLADA DE CORRECCIONES',table(['Iter.','Var.','Anterior','Valor GS','Corrección','Nuevo'],details),'RESULTADOS Y COMPROBACIÓN',('CRITERIO DE PARO CUMPLIDO. Revisar residuo.' if stopped else 'LÍMITE DE ITERACIONES ALCANZADO.'),f'Iteraciones: {k}\nResiduo inicial: {initial:.12f}']
    out += [f'{name} = {v:.12f}\n  Comprobación directa: {e:.12f}\n  Diferencia: {abs(v-e):.12f}' for name,v,e in zip(names,x,exact)]+[f'Residuo final max |b-Ax|: {res:.12f}']
    return '\n\n'.join(out)

def poly(c): return ' + '.join(f'({v})*x^{i}' for i,v in enumerate(c) if v) or '0'
def mul(c,z):
    r=[F(0)]*(len(c)+1)
    for i,v in enumerate(c): r[i]-=v*z; r[i+1]+=v
    return r

def interp(x,y,targets,mode,pairs):
    if len(x)!=len(y) or len(x)<2 or len(set(x))!=len(x) or len(x)>30: raise ValueError('Usa de 2 a 30 puntos con x distintos.')
    pts=sorted(zip(x,y)); x,y=map(list,zip(*pts)); out=['TABLA DE DATOS ORDENADOS POR X',table(['i','X','Y'],[[i+1,u,v] for i,(u,v) in enumerate(pts)])]; final=[]
    if mode=='Lagrange':
        c=[F(0)]*len(x); den=[]
        for i,u in enumerate(x):
            p=[F(1)]; d=F(1)
            for j,z in enumerate(x):
                if j!=i: p=mul(p,z); d*=u-z
            den.append(d); out.append(f'L{i+1}(x) = '+ ' * '.join(f'(x - ({z}))' for j,z in enumerate(x) if j!=i)+f' / ({d})')
            c=[v+y[i]*w/d for v,w in zip(c,p)]
        out+=['POLINOMIO EXPANDIDO EXACTO',poly(c)]
        for t in targets:
            rows=[]
            for i,u in enumerate(x):
                num=math.prod(t-z for j,z in enumerate(x) if j!=i); l=num/den[i]; rows.append([i+1,y[i],num,den[i],l,y[i]*l])
            v=sum(r[-1] for r in rows); out += [f'PROCEDIMIENTO PARA X={t}',table(['i','Yi','Numerador','Denominador','Li(x)','Yi*Li'],rows),f'SUMA DE TÉRMINOS = {v}']; final.append([t,v,float(v)])
    else:
        p=nums(pairs)
        if len(p)%2 or any(z.denominator!=1 or not 1<=z<=len(x) for z in p): raise ValueError('Indica pares de índices válidos, ej. 1 2 7 8.')
        for t in targets:
            for i,j in zip(p[::2],p[1::2]):
                i,j=int(i)-1,int(j)-1; u,v=(x,y) if mode=='Lineal Y dado X' else (y,x); delta=u[j]-u[i]
                if not delta: raise ValueError('Puntos seleccionados con denominador cero.')
                ratio=(t-u[i])/delta; ans=v[i]+ratio*(v[j]-v[i]); kind='INTERPOLACIÓN' if 0<=ratio<=1 else 'EXTRAPOLACIÓN'
                out += [f'P1=({x[i]}, {y[i]}); P2=({x[j]}, {y[j]})',f't = ({t} - ({u[i]})) / ({delta}) = {ratio}\nResultado = {v[i]} + ({ratio})*({v[j]-v[i]}) = {ans}\nTIPO: {kind}']; final.append([t,ans,f'{float(ans):.10g} ({kind})'])
    return '\n\n'.join(out+['TABLA FINAL DE RESULTADOS',table(['Consulta','Resultado exacto','Decimal / tipo'],final)])

def fit(x,y,d,mode):
    if len(x)!=len(y) or not 1<=d<=6 or len(set(x))<d+1: raise ValueError('Grado 1–6; requiere al menos grado+1 valores x distintos.')
    out=[]; summary=[]; n=len(x)
    for g in (range(1,d+1) if mode=='Todos hasta el grado' else [d]):
        a=[[sum(z**(i+j) for z in x) for j in range(g+1)] for i in range(g+1)]; b=[sum(w*z**i for z,w in zip(x,y)) for i in range(g+1)]; r,_=reduce(a,[[v] for v in b]); c=[v[0] for v in r]; rows=[]; s=F(0)
        f=lambda v:f'{float(v):.10g}'
        for z,w in zip(x,y):
            yc=sum(v*z**i for i,v in enumerate(c)); e=w-yc; s+=e*e; rows.append(list(map(f,[z,w,yc,e,e*e])))
        mean=sum(y)/n; sst=sum((v-mean)**2 for v in y); rmse=math.sqrt(float(s/n)); r2=f(1-s/sst) if sst else 'No definido'; summary.append([g,f(s),f(rmse),r2]); out += [f'POLINOMIO DE GRADO {g}', 'Yc = '+poly([f(v) for v in c]),'Coeficientes:\n'+'\n'.join(f'K{i} = {f(v)}' for i,v in enumerate(c)), 'Evaluación de puntos',table(['x','y','Yc','y-Yc','(y-Yc)^2'],rows),f'S = {f(s)}\nRMSE = {f(rmse)}\nR² = {r2}',f'MATRIZ NORMAL DE GRADO {g}',showmat([r+[v] for r,v in zip(a,b)],g+1)]
    return '\n\n'.join(out+['COMPARACIÓN',table(['Grado','S','RMSE','R²'],summary)])


# Interfaz web. Las funciones matemáticas anteriores se conservan sin cambios.
NAMES = ['Gauss-Jordan', 'Gauss-Seidel y relajaciones', 'Interpolación', 'Ajuste de curvas']

def go(page):
    st.session_state.page = page
    st.rerun()

def run_web():
    st.set_page_config(page_title='Métodos Numéricos | Proyecto final', page_icon='📐', layout='wide')
    st.markdown('''<style>
    .mn-header {background:#183b60;color:white;padding:22px;border-radius:10px;text-align:center;font-size:26px;font-weight:700;margin-bottom:20px;}
    .mn-cover {text-align:center;padding:24px 10px;line-height:1.8;}
    .mn-cover h2,.mn-cover h3 {color:#347fc4;}
    </style>''',unsafe_allow_html=True)
    if 'page' not in st.session_state:
        st.session_state.page='home'
    page=st.session_state.page
    title='PROYECTO FINAL' if page=='home' else 'Seleccione un método' if page=='menu' else NAMES[page]
    st.markdown(f'<div class="mn-header">{title}</div>',unsafe_allow_html=True)
    if page=='home':
        st.markdown('''<div class="mn-cover"><h2>UNIVERSIDAD AUTÓNOMA DE CHIHUAHUA</h2>
        <h3>Facultad de Ingeniería</h3><h1>Métodos Numéricos</h1>
        <p>Alumno: Mario Mendoza Anchondo</p><p>Profesor: Oscar Mauricio Borunda Carrasco</p>
        <p>Ingeniería en Ciencias de la Computación</p><p>Agosto – Diciembre 2026</p></div>''',unsafe_allow_html=True)
        _,mid,_=st.columns([1,2,1])
        if mid.button('Ingresar al programa',type='primary',use_container_width=True): go('menu')
        return
    if page=='menu':
        descriptions=['Sistemas lineales y matriz inversa.','Iteraciones, errores, residuos y relajación.','Lagrange, interpolación lineal y extrapolación.','Mínimos cuadrados, grados 1 a 6.']
        for i,(name,description) in enumerate(zip(NAMES,descriptions)):
            with st.container(border=True):
                st.subheader(name)
                st.write(description)
                if st.button('Abrir método',key=f'open{i}',use_container_width=True): go(i)
        if st.button('Regresar a presentación'): go('home')
        return
    i=page
    if st.button('← Regresar al menú'): go('menu')
    st.caption('Separa los números con espacios. Para decimales usa punto; también se aceptan fracciones como 1/3. En matrices escribe una fila por línea.')
    with st.form(f'form{i}'):
        if i in (0,1):
            raw_a=st.text_area('Matriz A (una fila por línea)',height=120,max_chars=4000)
            raw_b=st.text_input('Vector b',max_chars=1000)
            if i==0:
                mode=st.selectbox('Operación',['Inversa y solución','Sistema directo'])
            else:
                start=st.text_input('Valores iniciales (vacío = ceros)',max_chars=1000)
                c1,c2,c3=st.columns(3)
                tol=c1.number_input('Tolerancia',value=0.01,format='%.10g',min_value=0.0)
                limit=c2.number_input('Máximo de iteraciones',min_value=1,max_value=2000,value=100,step=1)
                omega=c3.number_input('Omega (1 = GS)',value=1.0,format='%.6f',min_value=0.0,max_value=2.0)
                criterion=st.selectbox('Criterio',['Error absoluto','Error relativo','Error %','Residuo absoluto','Residuo relativo'])
                reorder=st.selectbox('Reordenar ecuaciones',['Sí','No'])
        else:
            raw_x=st.text_input('Valores x',max_chars=4000)
            raw_y=st.text_input('Valores y',max_chars=4000)
            if i==2:
                targets=st.text_input('Consultas (separadas por espacios)',max_chars=1000)
                mode=st.selectbox('Método',['Lagrange','Lineal Y dado X','Lineal X dado Y'])
                pairs=st.text_input('Pares de índices (solo lineal)',max_chars=1000,help='Los índices corresponden a los puntos ordenados por x. Escribe dos por cada par.')
            else:
                degree=st.text_input('Grado máximo (1–6)',max_chars=2)
                mode=st.selectbox('Modo',['Todos hasta el grado','Solo grado seleccionado'])
        submitted=st.form_submit_button('Calcular',type='primary',use_container_width=True)
    if submitted:
        st.session_state.pop(f'result{i}',None)
        try:
            # Límites adicionales para una aplicación alojada y de acceso público.
            raw_fields=[raw_a,raw_b]+([start] if i==1 else []) if i in (0,1) else [raw_x,raw_y]+([targets,pairs] if i==2 else [])
            for raw in raw_fields:
                for token in raw.replace(',',' ').split():
                    if len(token)>50 or 'e' in token.lower():
                        raise ValueError('Usa números o fracciones de hasta 50 caracteres, sin notación científica.')
            if i==0: out=jordan(mat(raw_a),nums(raw_b),mode)
            elif i==1:
                a=mat(raw_a); out=seidel(a,nums(raw_b),nums(start) if start.strip() else [F(0)]*len(a),tol,int(limit),omega,criterion,reorder)
            elif i==2:
                q=nums(targets)
                if len(q)>30: raise ValueError('Máximo 30 consultas por cálculo.')
                if len(pairs.split())>60: raise ValueError('Máximo 30 pares de índices.')
                out=interp(nums(raw_x),nums(raw_y),q,mode,pairs)
            else:
                x,y=nums(raw_x),nums(raw_y)
                if len(x)>200 or len(y)>200: raise ValueError('Máximo 200 puntos por cálculo.')
                out=fit(x,y,int(degree),mode)
            st.session_state[f'result{i}']=out
        except (ValueError,ArithmeticError) as exc:
            st.error(f'Revisa los datos: {exc}')
    if f'result{i}' in st.session_state:
        out=st.session_state[f'result{i}']
        st.subheader('Procedimiento y resultados')
        st.code(out,language=None)
        st.download_button('Descargar resultados (.txt)',out,file_name='resultados.txt',mime='text/plain')

if __name__=='__main__':
    run_web()
