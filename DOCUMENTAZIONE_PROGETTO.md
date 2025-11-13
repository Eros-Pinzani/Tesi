# DOCUMENTAZIONE PROGETTO - Simulatore Robot con ICP e Odometria

## PANORAMICA DEL PROGETTO

Questo progetto implementa un **simulatore di robot mobile con sensore LiDAR**, che permette di:
1. Simulare diversi tipi di traiettorie (rettilinee, circolari, a 8, random walk)
2. Acquisire scansioni LiDAR dell'ambiente
3. Confrontare l'**odometria** (stima della posizione basata sui comandi del robot) con la **localizzazione basata su ICP** (Iterative Closest Point)
4. Visualizzare e analizzare i risultati attraverso grafici interattivi

---

## ARCHITETTURA DEL PROGETTO

Il progetto è organizzato in moduli con responsabilità ben definite:

```
src/
├── main.py                    # Punto di ingresso e orchestratore principale
├── robot.py                   # Modello cinematico del robot
├── simulator.py               # Esecuzione simulazioni
├── trajectory_generator.py    # Generazione traiettorie
├── environment.py             # Gestione ostacoli e ambiente
├── environment_presets.py     # Configurazioni predefinite ambienti
├── lidar.py                   # Simulazione sensore LiDAR
├── icp.py                     # Algoritmo ICP per localizzazione
├── icp_plots.py              # Visualizzazione risultati ICP
└── visualizer.py             # Visualizzazione traiettorie e scansioni
```

---

## COMPONENTI PRINCIPALI

### 1. ROBOT (robot.py)

**Scopo**: Rappresenta lo stato del robot e la sua cinematica.

**Stato del robot**:
- `x, y`: Posizione nel piano (metri)
- `theta`: Orientamento (radianti, normalizzato in [-π, π])
- `v`: Velocità lineare (m/s)
- `omega`: Velocità angolare (rad/s)

**Metodi principali**:
- `state()`: Restituisce `[x, y, theta]`
- `set_command(v, omega)`: Imposta i comandi di velocità
- `step(dt)`: **Integrazione numerica** con schema di Eulero esplicito:
  ```
  x_{k+1} = x_k + v_k * cos(theta_k) * dt
  y_{k+1} = y_k + v_k * sin(theta_k) * dt
  theta_{k+1} = theta_k + omega_k * dt
  ```
- `_normalize_angle()`: Normalizza theta in [-π, π]

---

### 2. SIMULATORE (simulator.py)

**Scopo**: Orchestrare l'esecuzione della simulazione del robot.

**Metodi principali**:
- `run_from_sequence(vs, omegas, dt)`: 
  - Applica sequenze di comandi (velocità lineari `vs` e angolari `omegas`)
  - Avanza la simulazione con passo temporale `dt`
  - Registra la **history** degli stati `(N+1, 3)` con stato iniziale
  - Registra i **commands** applicati `(N, 2)`
  
- `reset_robot(x, y, theta)`: Reinizializza il robot in una nuova posizione

**Output**: La traiettoria completa del robot (ground truth per confronti)

---

### 3. GENERATORE DI TRAIETTORIE (trajectory_generator.py)

**Scopo**: Fornire sequenze di comandi per diverse tipologie di movimento.

**Traiettorie disponibili**:

1. **`straight(v, T, dt)`**: Moto rettilineo uniforme
   - Velocità lineare costante `v`
   - Velocità angolare `omega = 0`

2. **`straight_var_speed(v_min, v_max, T, dt, phase)`**: Moto rettilineo con velocità variabile
   - Profilo sinusoidale: `v(t) = v_mid + v_amp * sin(2πt/T + phase)`
   - Sempre `omega = 0`

3. **`circle(v, radius, T, dt)`**: Traiettoria circolare a velocità costante
   - Relazione cinematica: `omega = v / radius`
   - Raggio costante

4. **`circle_var_speed(v_min, v_max, radius, T, dt, phase)`**: Cerchio con velocità variabile
   - `v(t)` sinusoidale
   - `omega(t) = v(t) / radius` per mantenere raggio costante

5. **`eight(v, radius, T, dt)`**: Traiettoria a forma di 8
   - Prima metà: rotazione oraria (`omega = +v/radius`)
   - Seconda metà: rotazione antioraria (`omega = -v/radius`)
   - Transizione smooth (15% del tempo) per evitare discontinuità

6. **`random_walk(v_mean, omega_std, T, dt, seed)`**: Esplorazione casuale
   - Velocità lineare costante
   - Velocità angolare: rumore gaussiano `omega ~ N(0, omega_std²)`

---

### 4. AMBIENTE (environment.py)

**Scopo**: Gestire gli ostacoli e i confini dell'ambiente usando la libreria Shapely.

**Struttura dati**:
- `obstacles`: Lista di geometrie Shapely (rettangoli, cerchi, poligoni, muri)
- `bounds`: Rettangolo che delimita l'area di simulazione
- `_union_cache`: Cache dell'unione di tutti gli ostacoli (per performance)

**Metodi per aggiungere ostacoli**:
- `add_rectangle(xmin, ymin, xmax, ymax)`: Rettangolo axis-aligned
- `add_circle(cx, cy, radius, resolution=32)`: Cerchio approssimato con poligono
- `add_polygon(vertices)`: Poligono generico da lista di vertici
- `add_wall(x0, y0, x1, y1, thickness)`: Muro sottile bufferizzato

**Metodi di utilità**:
- `obstacles_union()`: Restituisce unione di tutti gli ostacoli (ottimizzata con cache)
- `first_intersection_with_line(line)`: Trova il primo punto di intersezione tra una linea e gli ostacoli
  - Usato dal LiDAR per calcolare gli impatti
  - Restituisce il punto più vicino all'origine del raggio

**environment_presets.py**: 
- `setup_environment(histories)`: Crea ambiente con bounds adattivi e ostacoli standard
- `setup_environments_per_trajectory(histories, titles)`: Crea ambienti personalizzati per ogni traiettoria
  - Posiziona ostacoli evitando collisioni con il percorso
  - Usa buffer di sicurezza attorno alla traiettoria
  - Garantisce varietà di forme per feature ricche al LiDAR

---

### 5. LIDAR (lidar.py)

**Scopo**: Simulare un sensore LiDAR 2D.

**Parametri del sensore**:
- `n_rays`: Numero di raggi per scansione (default 360)
- `angle_span`: Ampiezza angolare totale (default 2π radianti = 360°)
- `r_max`: Portata massima dei raggi (metri)
- `angle_offset`: Offset angolare rispetto al robot
- `add_noise`: Flag per aggiungere rumore gaussiano
- `noise_std`: Deviazione standard del rumore (metri)

**Metodi principali**:

1. **`scan(robot_state, env, return_ranges)`**: Scansione completa
   - Input: `robot_state = [x, y, theta]`
   - Per ogni raggio:
     - Calcola angolo in frame mondo: `angle = theta + angle_offset + angolo_relativo`
     - Crea LineString da posizione robot fino a `r_max`
     - Trova intersezione con ostacoli usando `env.first_intersection_with_line()`
     - Se c'è impatto: registra distanza e punto
     - Se nessun impatto: punto a `r_max`
     - Opzionale: aggiunge rumore gaussiano alla distanza
   - Output: array `(n_rays, 2)` con coordinate mondo dei punti

2. **`scan_hits(robot_state, env, frame)`**: Solo punti di impatto reali
   - Filtra i raggi che hanno colpito ostacoli (esclude quelli a `r_max`)
   - `frame='world'`: coordinate globali (default per visualizzazione)
   - `frame='local'`: coordinate nel frame del sensore (**OBBLIGATORIO per ICP**)
     - Traslazione: `-(x, y)`
     - Rotazione: `-(theta + angle_offset)` per portare asse x in avanti
     - **Motivazione**: L'ICP confronta scansioni in frame locale perché deve stimare la trasformazione relativa tra due pose consecutive
   - Output: array `(N, 2)` con solo gli hit validi

3. **`scan_hits_indexed(robot_state, env, frame)`**: Hit con indici dei raggi
   - Come `scan_hits` ma restituisce anche gli indici dei raggi che hanno colpito
   - Utile per analisi successive

---

### 6. ICP - ITERATIVE CLOSEST POINT (icp.py)

**Scopo**: Implementare l'algoritmo ICP per stimare la trasformazione tra due scansioni LiDAR consecutive.

#### COSA FA L'ICP

L'ICP è un algoritmo iterativo che allinea due nuvole di punti (point clouds) trovando la rotazione `R` e traslazione `t` ottimali tali che:
```
target ≈ R * source + t
```

**IMPORTANTE - Perché usare frame locale per ICP:**

L'ICP **deve** ricevere le scansioni in frame locale (non world) perché:

1. **Obiettivo dell'ICP**: Stimare la trasformazione **relativa** tra due pose consecutive
   - Vogliamo trovare: "Come si è mosso il robot da k-1 a k?"
   - Non: "Dove sono i punti nel mondo?"

2. **Se usassimo frame world**:
   - Scan al tempo k-1: punti in coordinate globali rispetto alla posa (x₁, y₁, θ₁)
   - Scan al tempo k: punti in coordinate globali rispetto alla posa (x₂, y₂, θ₂)
   - I punti sarebbero già "allineati" al mondo, non tra loro
   - L'ICP cercherebbe di far combaciare punti che rappresentano ostacoli diversi in posizioni globali diverse

3. **Con frame locale**:
   - Scan al tempo k-1: punti centrati nell'origine del robot a k-1
   - Scan al tempo k: punti centrati nell'origine del robot a k
   - Entrambi rappresentano la **stessa scena** ma da prospettive leggermente diverse
   - L'ICP trova la trasformazione che allinea queste due prospettive → movimento del robot

**Esempio concreto**:
- Robot al tempo k-1: (5m, 3m, 30°) vede un muro a 2m davanti
- Robot al tempo k: (6m, 4m, 45°) vede lo stesso muro
- Frame world: i punti del muro sono in posizioni globali completamente diverse
- Frame local: in entrambi i casi il muro è a ~2m dall'origine del sensore
- ICP in locale → trova la trasformazione che allinea le due viste del muro

**Algoritmo**:
1. **Inizializzazione**: Applica trasformazione iniziale (da odometria o identità)
2. **Iterazioni**:
   - Trova corrispondenze (nearest neighbors) tra source trasformato e target
   - Calcola trasformazione ottimale con SVD
   - Applica trasformazione
   - Calcola RMSE (Root Mean Square Error)
   - Verifica convergenza (se variazione RMSE < tolleranza → stop)
3. **Output**: Trasformazione cumulativa finale

**Organizzazione metodi**:

Il modulo ICP è organizzato in **tre categorie** di funzioni:

---

#### A) METODI DI BASE ICP (funzionano sia con che senza odometria)

Questi sono i metodi core dell'algoritmo ICP, indipendenti dall'inizializzazione:

1. **`find_nearest_neighbors(source, target, max_distance)`**:
   - Per ogni punto in source, trova il punto più vicino in target
   - Filtra coppie con distanza > `max_distance`
   - Output: due array di punti matched
   - **Usato da**: Entrambe le varianti (con e senza odometria)

2. **`compute_transformation_svd(source, target)`**:
   - Calcola R e t ottimali usando Singular Value Decomposition
   - Centra i punti rispetto ai centroidi
   - Matrice di covarianza: `H = target_centered^T * source_centered`
   - SVD: `H = U * Σ * V^T`
   - Rotazione: `R = U * V^T`
   - Assicura det(R) = +1 (rotazione propria)
   - Traslazione: `t = centroid_target - R * centroid_source`
   - **Usato da**: Entrambe le varianti (con e senza odometria)

3. **`compute_rmse(source, target, R, t)`**:
   - Calcola Root Mean Square Error dopo trasformazione
   - Formula: `RMSE = sqrt(mean(||R*source + t - target||²))`
   - **Usato da**: Entrambe le varianti (con e senza odometria)

4. **`icp(source, target, init_R, init_t, max_iterations, tolerance, max_correspondence_distance)`**:
   - **Funzione principale ICP** (algoritmo generico)
   - **INPUT FONDAMENTALE**: `source` e `target` sono **SEMPRE scansioni LiDAR** (nuvole di punti)
   - Parametri `init_R` e `init_t` determinano solo il punto di partenza:
     - Se `None` → **ICP RAW** (senza odometria, parte da identità)
     - Se forniti → **ICP FILTRATO** (con odometria, parte da stima odometrica)
   - **IMPORTANTE**: In entrambi i casi, l'algoritmo usa **SOLO i dati LiDAR** per iterare e trovare la soluzione
   - Algoritmo iterativo:
     1. Applica trasformazione iniziale
     2. Loop fino a convergenza:
        - Trova corrispondenze nearest neighbor
        - Calcola trasformazione ottimale (SVD)
        - Aggiorna trasformazione cumulativa
        - Calcola RMSE
        - Verifica convergenza (|RMSE_prev - RMSE| < tolerance)
   - Output: dizionario con R, t, RMSE, numero iterazioni, errori per iterazione, ecc.
   - **Usato da**: Wrapper `run_icp_pair` per entrambe le modalità

---

#### B) METODI PER ODOMETRIA (solo per inizializzazione ICP)

Questi metodi sono usati **esclusivamente** per calcolare la stima iniziale da odometria:

**IMPORTANTE**: L'odometria fornisce solo il **punto di partenza** per l'ICP. L'algoritmo ICP usa **sempre** le scansioni LiDAR per trovare la trasformazione ottimale.

1. **`compute_relative_transform_from_odometry(prev_pose, curr_pose)`**:
   - **Scopo**: Calcola trasformazione relativa tra due pose consecutive usando SOLO l'odometria (NON le scansioni)
   - **Quando viene usato**: Solo per inizializzare R e t prima di passare i dati all'ICP
   - Input: `prev_pose = [x, y, theta]` al tempo k-1, `curr_pose` al tempo k (solo pose, no LiDAR)
   - Output: `R` (matrice 2x2 di rotazione), `t` (vettore 2D di traslazione) come stima iniziale
   - **Formula**:
     ```
     # Differenza angolare
     d_theta = theta_curr - theta_prev
     
     # Matrice di rotazione relativa
     R = [[cos(d_theta), -sin(d_theta)],
          [sin(d_theta),  cos(d_theta)]]
     
     # Traslazione nel frame locale di k-1:
     dx_world = x_curr - x_prev
     dy_world = y_curr - y_prev
     t_x = cos(theta_prev) * dx_world + sin(theta_prev) * dy_world
     t_y = -sin(theta_prev) * dx_world + cos(theta_prev) * dy_world
     t = [t_x, t_y]
     ```
   - **Usato da**: Solo `run_icp_pair` per inizializzare ICP Filtrato
   - **NON usato da**: ICP RAW

---

#### C) WRAPPER PRINCIPALE (esegue ENTRAMBE le varianti)

1. **`run_icp_pair(prev_pose, curr_pose, src_local, tgt_local, ...)`**:
   - **Scopo**: Wrapper ad alto livello che esegue e confronta entrambe le modalità ICP
   - **Flusso di esecuzione**:
     
     **Passo 1**: Calcola inizializzazione da odometria
     ```python
     R_odom, t_odom = compute_relative_transform_from_odometry(prev_pose, curr_pose)
     ```
     
     **Passo 2**: Esegue **ICP FILTRATO** (con odometria)
     ```python
     result_filtered = icp(
         src_local, tgt_local,
         init_R=R_odom,      # ← Inizializzazione da odometria
         init_t=t_odom,      # ← Inizializzazione da odometria
         max_iterations=50,
         tolerance=1e-6,
         max_correspondence_distance=0.5
     )
     ```
     
     **Passo 3**: Esegue **ICP RAW** (senza odometria)
     ```python
     result_raw = icp(
         src_local, tgt_local,
         init_R=None,        # ← Parte da identità
         init_t=None,        # ← Parte da zero
         max_iterations=50,
         tolerance=1e-6,
         max_correspondence_distance=0.5
     )
     ```
     
     **Passo 4**: Restituisce dizionario con entrambi i risultati
     ```python
     return {
         'none': result_filtered,      # ICP con odometria
         'raw_none': result_raw,        # ICP senza odometria
         'gt_R': R_odom,                # Ground truth da odometria
         'gt_t': t_odom,
         'src_local': src_local,
         'tgt_local': tgt_local,
         ...
     }
     ```

---

#### RIEPILOGO: Quale metodo usa cosa?

| Metodo | Usa Odometria? | Funzione Chiamata | Inizializzazione |
|--------|----------------|-------------------|------------------|
| **ICP FILTRATO** (`none`) | ✅ SÌ | `icp(init_R=R_odom, init_t=t_odom)` | Da odometria |
| **ICP RAW** (`raw_none`) | ❌ NO | `icp(init_R=None, init_t=None)` | Identità/Zero |
| `compute_relative_transform_from_odometry` | ✅ SÌ (è l'odometria) | - | - |
| `find_nearest_neighbors` | ❌ NO | Usata da `icp()` | N/A |
| `compute_transformation_svd` | ❌ NO | Usata da `icp()` | N/A |
| `compute_rmse` | ❌ NO | Usata da `icp()` | N/A |

---

#### DIFFERENZE CHIAVE tra ICP Filtrato e ICP RAW

**IMPORTANTE**: Entrambe le varianti usano **SEMPRE** le scansioni LiDAR come dati primari!

| Aspetto | ICP FILTRATO (con odometria) | ICP RAW (senza odometria) |
|---------|------------------------------|---------------------------|
| **Scansioni LiDAR** | ✅ USA (source, target) | ✅ USA (source, target) |
| **Odometria** | ✅ Solo per inizializzazione | ❌ Non usata |
| **Inizializzazione** | R e t da odometria | R = I (identità), t = 0 |
| **Prima iterazione** | Parte vicino alla soluzione | Parte da zero |
| **Iterazioni successive** | Solo dati LiDAR | Solo dati LiDAR |
| **Convergenza** | Più veloce (5-15 iterazioni) | Più lenta (15-30 iterazioni) |
| **Robustezza** | Alta (meno minimi locali) | Media (rischio minimi locali) |
| **Precisione finale** | Alta (se odometria buona) | Variabile (dipende da geometria) |
| **Quando fallisce** | Se odometria molto sbagliata | Se scena simmetrica/povera |
| **Uso tipico** | Localizzazione incrementale | Localizzazione globale, relocalization |

**In sintesi**:
- L'odometria fornisce solo una **"prima ipotesi"** migliore
- Entrambi gli algoritmi **raffinano sempre** usando le scansioni LiDAR
- Il risultato finale è **sempre basato sui dati LiDAR**, non sull'odometria

---

### 7. ODOMETRIA - COME FUNZIONA

#### DEFINIZIONE
L'**odometria** è la stima della posizione e orientamento del robot basata sull'integrazione dei comandi di velocità nel tempo.

#### PRINCIPIO DI FUNZIONAMENTO

1. **Stato del robot**: `[x, y, theta]`
   - `x, y`: posizione nel piano
   - `theta`: orientamento (angolo rispetto all'asse x)

2. **Comandi**: `[v, omega]`
   - `v`: velocità lineare (m/s)
   - `omega`: velocità angolare (rad/s)

3. **Modello cinematico** (integrazione di Eulero):
   ```
   x(t+dt) = x(t) + v(t) * cos(theta(t)) * dt
   y(t+dt) = y(t) + v(t) * sin(theta(t)) * dt
   theta(t+dt) = theta(t) + omega(t) * dt
   ```

4. **Processo**:
   - Ad ogni step temporale `dt`:
     - Leggi velocità `v` e `omega` dai comandi (o dai motori)
     - Aggiorna posizione usando le equazioni sopra
     - Normalizza `theta` in [-π, π]
     - Salva nuovo stato

#### CARATTERISTICHE

**Vantaggi**:
- Molto veloce (solo calcoli aritmetici)
- Non richiede sensori esterni
- Alta frequenza di aggiornamento

**Svantaggi**:
- **Drift**: errori si accumulano nel tempo
- Sensibile a:
  - Slittamento delle ruote
  - Imperfezioni del terreno
  - Errori di calibrazione
  - Errori numerici nell'integrazione

#### NEL PROGETTO

L'odometria è implementata in:
- **robot.py**: metodo `step(dt)` che integra le equazioni
- **simulator.py**: applica comandi sequenzialmente salvando la storia
- **icp.py**: `compute_relative_transform_from_odometry()` calcola la trasformazione relativa tra due pose consecutive

L'odometria fornisce:
1. **Ground truth** (nelle simulazioni senza rumore)
2. **Inizializzazione per ICP** (prima stima della trasformazione)
3. **Baseline per confronto** con localizzazione basata su sensori

---

### 8. VISUALIZZATORE (visualizer.py)

**Scopo**: Visualizzazione interattiva e salvataggio immagini.

**Funzionalità principali**:

1. **`draw_robot(ax, state, robot_radius, ...)`**:
   - Disegna robot come rettangolo orientato con:
     - Corpo rettangolare (ruotato secondo theta)
     - 4 ruote ai lati
     - Pallino centrale
     - Freccia di orientamento
   - Scala dinamica basata su estensione traiettoria
   - Colori: verde=partenza, rosso=arrivo, blu=intermedi

2. **`plot_trajectory(history, show_orient_every, title, save_path, environment, fit_to)`**:
   - Plot statico di singola traiettoria
   - Disegna ambiente (ostacoli) in background
   - Linea nera per percorso
   - Robot a intervalli regolari
   - Salva PNG in `img/trajectories/`

3. **`show_trajectories_carousel(histories, titles, ...)`**:
   - **Viewer interattivo** con 3 pannelli:
     - **Reale**: Traiettoria ground truth
     - **ICP RAW**: Localizzazione ICP senza inizializzazione odometrica
     - **ICP Filtrato**: Localizzazione ICP con inizializzazione da odometria
   - **Controlli**:
     - Pulsante "Precedente" / "Successivo": naviga tra traiettorie
     - Pulsante "Play/Pausa": animazione temporale
     - Timer: avanza frame automaticamente
   - **Features**:
     - Visualizzazione progressiva dei robot durante playback
     - Pannello info (tempo, velocità, posa)
     - Rilevamento collisioni con stop automatico
     - Calcolo ICP on-demand con caching
     - Visualizzazione raggi LiDAR (opzionale)

4. **`save_trajectories_images(histories, titles, ...)`**:
   - Salva PNG per tutte le traiettorie
   - Robot statici sovrapposti
   - Callback di progresso

5. **`save_lidar_scans_images(history, title, lidar, environment, dt, interval_s, ...)`**:
   - Salva immagini delle scansioni LiDAR a intervalli regolari
   - Cartella: `img/scans/{nome_traiettoria}/`
   - Mostra:
     - Ambiente (bounds e ostacoli) in background
     - Raggi LiDAR (linee rosse semi-trasparenti)
     - Punti di impatto (scatter rossi)
     - **Robot nella posa corrente** (rettangolo blu con freccia di orientamento)
   - Scala del robot adattiva all'estensione dell'ambiente

6. **`cleanup_output_images(subfolders, remove_root)`**:
   - Pulisce cartelle di output prima di nuovo run
   - Evita accumulo di immagini obsolete

---

### 9. PLOT ICP (icp_plots.py)

**Scopo**: Funzioni specializzate per visualizzare analisi ICP.

**Funzioni**:

1. **`save_concept_correspondences(res, title, out_path, max_lines)`**:
   - Disegna schema concettuale delle corrispondenze ICP
   - Linee grigie collegano punti source a nearest neighbors in target
   - Limita numero di linee per leggibilità

2. **`save_alignment_overlays(res, title, out_path)`**:
   - Overlay finale: target + source trasformato
   - Confronta risultati ICP filtrato vs RAW

3. **`save_convergence_curves(res, title, out_path)`**:
   - Plot RMSE vs numero iterazioni
   - Mostra velocità di convergenza

4. **`save_motion_arrows(res, title, out_path)`**:
   - Frecce che rappresentano trasformazione stimata
   - Mostra Δx, Δy e angolo di rotazione

5. **`save_raw_vs_filtered(res, title, out_path)`**:
   - Confronto diretto RAW vs Filtrato su stessi assi

---

### 10. MAIN (main.py)

**Scopo**: Orchestratore principale che coordina tutte le componenti.

**Struttura workflow**:

1. **Setup iniziale**:
   - Parsing argomenti linea di comando
   - Configurazione logging su file
   - Pulizia cartelle output

2. **Definizione traiettorie**:
   - 6 tipologie predefinite (rettilinea costante/variabile, circolare costante/variabile, a 8, random walk)
   - Parametri: velocità, durata, dt, raggio

3. **Simulazione**:
   - Per ogni traiettoria:
     - Genera comandi con `TrajectoryGenerator`
     - Esegue simulazione con `Simulator`
     - Registra history (ground truth)

4. **Setup ambienti**:
   - `setup_environments_per_trajectory()` crea ambienti personalizzati
   - Ostacoli posizionati strategicamente evitando collisioni

5. **Scansioni LiDAR**:
   - Crea sensore LiDAR con parametri configurati
   - Per ogni traiettoria:
     - Salva scansioni cartesiane a intervalli
     - Salva scansioni polari (opzionale)

6. **Analisi ICP**:
   - Per ogni traiettoria:
     - **Scansioni consecutive in frame locale** (FONDAMENTALE)
       - `prev_local = lidar.scan_hits(prev_pose, env, frame='local')`
       - `curr_local = lidar.scan_hits(curr_pose, env, frame='local')`
     - Esegue ICP con e senza inizializzazione
     - Calcola traiettorie stimate
     - Salva plot di analisi (convergenza, overlay, frecce, ecc.)
     - **Stampa dati numerici nel log** (Reali, ICP, RAW per ogni step)

7. **Visualizzazione**:
   - Salva immagini statiche di tutte le traiettorie
   - Lancia viewer interattivo carousel (default: attivo, disattivabile con `--skip-viewer`)

8. **Output**:
   - `img/trajectories/`: Plot traiettorie
   - `img/scans/`: Scansioni LiDAR cartesiane
   - `img/scans_polar/`: Scansioni polari
   - `img/icp/`: Analisi ICP (convergenza, overlay, frecce)
   - `logs/`: File di log con timestamp (contengono anche tutti i dati numerici ICP)

**Argomenti da linea di comando**:
- `--skip-viewer`: Salta il viewer interattivo (default: viewer attivo)
- `--skip-icp`: Non esegue l'analisi ICP (default: ICP attivo)
- `--skip-collision`: Salta il calcolo delle collisioni (default: attivo)
- `--viewer-mode {carousel,grid}`: Modalità viewer (default: carousel)
- `--icp-step N`: Intervallo tra scan per ICP (default: 1 = consecutivi)
- `--scan-interval SECONDI`: Intervallo temporale per salvataggio scansioni (default: 1.0s)
- `--viewer-lidar-every N`: Aggiorna visualizzazione LiDAR ogni N frame nel viewer (default: 4)
- `--viewer-log-align-world`: Allinea traiettorie ricostruite dal log al frame mondo
- `--quiet`: Sopprime stampe durante salvataggio immagini
- Parametri LiDAR: `--lidar-rays N`, `--lidar-range M`, ecc.

**Comportamento default** (senza argomenti):
- ✅ Esegue analisi ICP completa
- ✅ Salva tutte le immagini (traiettorie, analisi ICP)
- ✅ Salva scansioni LiDAR cartesiane e polari per tutte le traiettorie
- ✅ Apre viewer interattivo in modalità carousel

---

## OUTPUT E IMMAGINI GENERATE

Ad ogni esecuzione della simulazione, il sistema genera automaticamente diverse tipologie di immagini organizzate in cartelle specifiche. Ecco una guida completa per comprendere ogni tipo di output.

### STRUTTURA CARTELLE OUTPUT

```
img/
├── trajectories/           # Traiettorie complete con robot
├── scans/                 # Scansioni LiDAR cartesiane
│   ├── rettilinea_v_costante/
│   ├── rettilinea_v_variabile/
│   ├── circolare_v_costante/
│   ├── circolare_v_variabile/
│   ├── traiettoria_a_8/
│   └── random_walk/
├── scans_polar/           # Scansioni LiDAR in coordinate polari
│   └── [stessa struttura di scans/]
└── icp/                   # Analisi ICP
    ├── concept/           # Schemi concettuali corrispondenze
    ├── overlays/          # Sovrapposizioni allineamenti
    ├── convergence/       # Curve di convergenza
    ├── arrows/            # Frecce movimento stimato
    └── raw_vs_filtered/   # Confronti RAW vs Filtrato
```

---

### 1. IMMAGINI TRAIETTORIE (`img/trajectories/`)

**Formato**: `{nome_traiettoria}_{timestamp}.png`  
**Esempio**: `rettilinea_v_costante_20251113-143052.png`

**Contenuto**:
- **Linea nera**: Percorso completo della traiettoria
- **Ambiente**: Bounds (contorno grigio) e ostacoli (grigio riempito)
- **Robot a intervalli**:
  - **Verde**: Posizione iniziale (t=0)
  - **Blu**: Posizioni intermedie (ogni N step)
  - **Rosso**: Posizione finale
- **Freccia arancione**: Direzione di marcia del robot
- **Ruote**: 4 cerchi bianchi con bordo nero
- **Pallino centrale**: Arancione (verde all'inizio, rosso alla fine)

**Scopo**: Visualizzare la traiettoria completa percorsa dal robot, verificare che non ci siano collisioni e comprendere il tipo di movimento.

**6 traiettorie generate**:
1. `rettilinea_v_costante`: Moto rettilineo uniforme
2. `rettilinea_v_variabile`: Linea retta con velocità sinusoidale
3. `circolare_v_costante`: Cerchio a velocità costante
4. `circolare_v_variabile`: Cerchio con velocità variabile
5. `traiettoria_a_8`: Figura a otto con transizione smooth
6. `random_walk`: Movimento casuale esplorativo

---

### 2. SCANSIONI LIDAR CARTESIANE (`img/scans/{traiettoria}/`)

**Formato**: `{traiettoria}_t{tempo}s_points_{timestamp}.png`  
**Esempio**: `circolare_v_costante_t2.00s_points_20251113-143055.png`

**Contenuto**:
- **Ambiente**: Bounds e ostacoli in background
- **Robot (BLU)**:
  - Corpo: rettangolo blu semi-trasparente
  - Freccia arancione: direzione sensore
  - 4 ruote: cerchi bianchi con bordo nero
  - Centro: pallino arancione
- **Raggi LiDAR**: Linee rosse semi-trasparenti (α=0.35) dal centro robot ai punti
- **Punti di impatto**: Scatter rossi (dove i raggi colpiscono ostacoli)

**Frequenza**: Immagini salvate a intervalli regolari (default: ogni 1 secondo di simulazione)

**Scopo**: 
- Visualizzare cosa "vede" il sensore LiDAR in un dato istante
- Verificare la correttezza delle misure di distanza
- Comprendere la densità e distribuzione dei punti percepiti
- Analizzare la copertura del sensore rispetto agli ostacoli

**Dettagli tecnici**:
- Scala robot adattiva all'ambiente (robot_radius ≈ 1.5% dell'estensione)
- Coordinate mondo (frame='world')
- Solo raggi con impatto reale (no r_max)

---

### 3. SCANSIONI LIDAR POLARI (`img/scans_polar/{traiettoria}/`)

**Formato**: `{traiettoria}_polar_t{tempo}s_{timestamp}.png`  
**Esempio**: `traiettoria_a_8_polar_t4.50s_20251113-143101.png`

**Contenuto**:
- **Grafico r(θ)**: Distanza (r) in funzione dell'angolo (θ)
- **Asse X**: Angolo in gradi [0° - 360°]
- **Asse Y**: Distanza in metri [0 - r_max]
- **Punti BLU**: Hit reali (raggi che hanno colpito ostacoli)
- **Punti GRIGI** (opzionale): Miss (raggi a r_max senza impatto)
- **Griglia**: Sfondo con griglia per lettura valori

**Scopo**:
- Rappresentazione alternativa delle scansioni (formato tipico dei sensori LiDAR)
- Analisi quantitativa delle distanze misurate
- Identificazione di pattern geometrici (muri → linee orizzontali, angoli → discontinuità)
- Verifica copertura angolare del sensore

**Interpretazione**:
- **Linea orizzontale**: Muro parallelo al robot
- **Curva smooth**: Superficie curva (cerchio)
- **Discontinuità verticali**: Spigoli, angoli
- **Vuoti**: Direzioni senza ostacoli (fino a r_max)

---

### 4. ANALISI ICP - CORRISPONDENZE (`img/icp/concept/`)

**Formato**: `concept_{nome_traiettoria}.png`  
**Esempio**: `concept_circolare_v_variabile.png`

**Contenuto**:
- **Punti BLU**: Target (scansione al tempo k-1)
- **Punti ROSSI**: Source (scansione al tempo k)
- **Linee GRIGIE**: Corrispondenze nearest neighbor
  - Collegano ogni punto source al suo punto più vicino in target
  - Massimo 120 linee per leggibilità

**Scopo**:
- Visualizzare il concetto di "corrispondenze" nell'ICP
- Vedere come l'algoritmo accoppia i punti tra due scansioni
- Identificare visivamente se le corrispondenze sono sensate

**Note**: 
- Rappresenta solo un subset delle coppie (per non sovraffollare il grafico)
- Scansioni in frame locale

---

### 5. ANALISI ICP - OVERLAY ALLINEAMENTI (`img/icp/overlays/`)

**Formato**: `overlay_{nome_traiettoria}.png`  
**Esempio**: `overlay_rettilinea_v_costante.png`

**Contenuto**:
- **Punti NERI**: Target (riferimento, tempo k-1)
- **Punti ROSSI**: Source trasformato con ICP Filtrato (con odometria)
- **Punti ARANCIONI**: Source trasformato con ICP RAW (senza odometria)

**Scopo**:
- Confrontare qualità dell'allineamento finale
- Verificare sovrapposizione tra target e source trasformato
- Valutare l'effetto dell'inizializzazione da odometria

**Interpretazione**:
- **Sovrapposizione perfetta** (rosso/arancione su nero): ICP ha funzionato bene
- **Offset residuo**: Errore di allineamento (ICP non convergente o scena ambigua)
- **Differenza rosso-arancione**: Beneficio dell'inizializzazione odometrica

---

### 6. ANALISI ICP - CONVERGENZA (`img/icp/convergence/`)

**Formato**: `convergence_{nome_traiettoria}.png`  
**Esempio**: `convergence_random_walk.png`

**Contenuto**:
- **Grafico RMSE vs Iterazioni**
- **Linea continua**: ICP Filtrato (con odometria)
- **Linea tratteggiata**: ICP RAW (senza odometria)
- **Asse X**: Numero iterazione ICP
- **Asse Y**: RMSE (Root Mean Square Error) in metri

**Scopo**:
- Analizzare velocità di convergenza dell'algoritmo
- Confrontare prestazioni con/senza inizializzazione
- Verificare stabilità (curva deve decrescere monotonicamente)

**Interpretazione**:
- **Curva ripida all'inizio**: Convergenza rapida (buona inizializzazione)
- **Plateau finale**: Convergenza raggiunta
- **Oscillazioni**: Possibile instabilità (rare con questo algoritmo)
- **ICP Filtrato converge più velocemente**: Evidenzia beneficio odometria

**Valori tipici**:
- RMSE iniziale: 0.1 - 1.0 m
- RMSE finale: 0.001 - 0.01 m
- Iterazioni: 5-30 (convergenza tipica < 20)

---

### 7. ANALISI ICP - FRECCE MOVIMENTO (`img/icp/arrows/`)

**Formato**: `arrows_{nome_traiettoria}.png`  
**Esempio**: `arrows_traiettoria_a_8.png`

**Contenuto**:
- **Origine (0,0)**: Punto di partenza
- **Frecce colorate**: Vettori di spostamento stimati
  - **Freccia ROSSA**: ICP Filtrato (con odometria)
  - **Freccia ARANCIONE**: ICP RAW (senza odometria)
- **Legenda**: Angolo di rotazione α in gradi

**Scopo**:
- Visualizzare la trasformazione stimata come vettore di movimento
- Confrontare direzione e magnitudine dello spostamento
- Verificare coerenza dell'angolo di rotazione stimato

**Interpretazione**:
- **Lunghezza freccia**: Distanza percorsa (√(Δx² + Δy²))
- **Direzione freccia**: Direzione del movimento nel frame locale
- **Angolo α**: Rotazione del robot (+ = antiorario, - = orario)

**Esempio lettura**:
```
ICP Filtrato: Δx=+0.15m, Δy=-0.02m, α=+5.2°
→ Robot si è mosso 15cm avanti, 2cm a destra, ruotato 5.2° antiorario
```

---

### 8. ANALISI ICP - RAW VS FILTRATO (`img/icp/raw_vs_filtered/`)

**Formato**: `raw_vs_filtered_{nome_traiettoria}.png`  
**Esempio**: `raw_vs_filtered_circolare_v_costante.png`

**Contenuto**:
- **Punti NERI**: Target (scansione k-1, riferimento)
- **Punti ARANCIONI**: Source trasformato RAW (senza inizializzazione)
- **Punti ROSSI**: Source trasformato Filtrato (con inizializzazione odometria)

**Scopo**:
- Confronto diretto delle due modalità ICP
- Evidenziare miglioramento dato dall'odometria
- Identificare casi critici (simmetrie, pochi feature)

**Interpretazione**:
- **Rosso sovrapposto a nero, arancione sfalsato**: Odometria cruciale
- **Rosso e arancione entrambi su nero**: Scena ricca di feature (ICP funziona anche senza odometria)
- **Entrambi sfalsati**: Possibile fallimento ICP (geometria ambigua)

---

### 9. LOG TESTUALI (`logs/`)

**Formato**: `run_output_{timestamp}.txt`  
**Esempio**: `run_output_20251113-143052.txt`

**Contenuto**:
- Output completo della console durante l'esecuzione
- Timestamp e parametri di avvio
- Progressi delle varie fasi (simulazione, ICP, salvataggio)
- Eventuali warning o errori
- **Dati numerici ICP completi per ogni passo**:
  - `Reali:` Δx, Δy, α da odometria (ground truth)
  - `ICP:` Δx, Δy, α stimati con ICP Filtrato (con odometria)
  - `RAW:` Δx, Δy, α stimati con ICP RAW (senza odometria)
  - RMSE, numero iterazioni, convergenza
  
**Esempio formato dati ICP nel log**:
```
CASO 3: circolare_v_costante
Step k=1:
  Reali:  Δx=+0.150 m, Δy=-0.020 m, α=+3.56 deg
  ICP:    Δx=+0.148 m, Δy=-0.021 m, α=+3.54 deg  [RMSE: 0.0042m, iter: 12]
  RAW:    Δx=+0.151 m, Δy=-0.019 m, α=+3.58 deg  [RMSE: 0.0051m, iter: 18]
```

**Scopo**:
- Debugging e tracciamento esecuzione
- Riproducibilità (parametri usati)
- **Analisi quantitativa completa** (i log contengono tutti i dati numerici ICP)
- Ricostruzione traiettorie ICP dal log (per viewer interattivo)
- Confronto numerico tra odometria e ICP
- Analisi post-mortem in caso di problemi

**Nota importante**: 
I dati numerici ICP **non sono più salvati in JSON** ma sono contenuti direttamente nel log testuale, rendendo più semplice la consultazione e l'analisi.

---

### RIEPILOGO UTILIZZO IMMAGINI E OUTPUT

| Tipo Output | Quando usarlo | Informazione Principale |
|-------------|---------------|-------------------------|
| **Traiettorie** | Presentazione, report | Percorso completo e movimento |
| **Scans cartesiane** | Debug LiDAR, verifica percezione | Cosa vede il sensore |
| **Scans polari** | Analisi quantitativa distanze | Profilo radiale ostacoli |
| **Concept ICP** | Spiegazione algoritmo | Come funziona l'accoppiamento |
| **Overlay** | Valutazione qualità ICP | Bontà allineamento |
| **Convergenza** | Analisi prestazioni | Velocità e stabilità ICP |
| **Arrows** | Visualizzazione movimento | Trasformazione stimata |
| **Raw vs Filtered** | Confronto metodi | Beneficio odometria |
| **Log testuali** | Analisi numerica, debugging | Dati esatti ICP + tracciamento completo |

---

### SUGGERIMENTI PER LA PRESENTAZIONE

**Per mostrare il funzionamento del LiDAR**:
→ Usa scansioni cartesiane con robot visibile

**Per spiegare l'ICP**:
→ Sequenza: Concept → Convergenza → Overlay

**Per dimostrare l'importanza dell'odometria**:
→ Usa Raw vs Filtered + grafici convergenza

**Per analisi quantitativa**:
→ Usa scansioni polari + dati dai log testuali

**Per overview generale**:
→ Usa immagini traiettorie complete

**Per debugging o analisi numerica dettagliata**:
→ Consulta i log testuali in `logs/` (contengono tutti i dati ICP per ogni step)

---

## FLUSSO DI ESECUZIONE COMPLETO

1. **Inizializzazione**: Parse argomenti, setup logging, pulizia output
2. **Generazione traiettorie**: 6 traiettorie con comandi (v, omega)
3. **Simulazione dinamica**: Integrazione cinematica → history (ground truth)
4. **Setup ambienti**: Creazione ostacoli personalizzati per ogni traiettoria
5. **Acquisizione dati LiDAR**: Scansioni a intervalli regolari
6. **Localizzazione ICP**: 
   - Confronto scan consecutivi
   - Stima trasformazione con/senza odometria
   - Ricostruzione traiettoria incrementale
7. **Analisi e confronto**:
   - Ground truth (simulatore) vs Odometria vs ICP
   - Metriche: RMSE, numero iterazioni, convergenza
8. **Visualizzazione**: Plot statici + viewer interattivo
9. **Salvataggio**: Immagini, log testuali con dati numerici

---

## CONFRONTO TRA METODI DI LOCALIZZAZIONE

| Aspetto | Odometria | ICP RAW | ICP Filtrato |
|---------|-----------|---------|--------------|
| **Input** | Comandi motori | Scan LiDAR | Scan LiDAR + Odometria |
| **Frequenza** | Alta (ogni dt) | Media (ogni scan) | Media (ogni scan) |
| **Errore drift** | Alto (accumula) | Basso (correzione) | Molto basso |
| **Tempo calcolo** | Bassissimo | Alto | Alto |
| **Inizializzazione** | Posa iniziale | Identità | Odometria |
| **Robustezza** | Bassa | Media | Alta |
| **Convergenza ICP** | N/A | Più lenta | Più veloce |

---

## LIBRERIE UTILIZZATE

- **NumPy**: Calcolo numerico, array, trigonometria
- **Matplotlib**: Visualizzazione grafici e animazioni
- **Shapely**: Geometria computazionale (ostacoli, intersezioni)
- **SciPy**: SVD per ICP
- **tqdm**: Barre di progresso

---

## CONCETTI CHIAVE PER LA PRESENTAZIONE

### 1. ODOMETRIA
- Stima posizione integrando comandi di velocità
- Veloce ma soggetta a drift
- Usata come inizializzazione per ICP

### 2. LIDAR
- Sensore che misura distanze con raggi laser
- 360 raggi a 360° attorno al robot
- Rileva ostacoli fino a portata massima (6m)
- Output: nuvola di punti (point cloud)

### 3. ICP (Iterative Closest Point)
- Allinea due nuvole di punti consecutive
- Trova rotazione e traslazione ottimali
- Corregge drift dell'odometria
- Migliora con buona inizializzazione

### 4. FRAME DI RIFERIMENTO
- **Mondo (world)**: Sistema globale fisso
  - Usato per: visualizzazione, traiettorie ground truth, ostacoli
  - Origine: punto fisso arbitrario dell'ambiente
- **Locale (local)**: Sistema solidale al robot (x avanti, origine al robot)
  - **Usato per ICP**: Le scansioni consecutive devono essere in frame locale
  - Perché local per ICP: L'algoritmo stima la trasformazione relativa (ΔR, Δt) tra due pose consecutive. Se le scansioni fossero in world, l'ICP cercherebbe di allineare punti già in posizioni globali diverse, rendendo impossibile stimare il movimento del robot
  - Esempio: Se al tempo k-1 il robot è in (5, 3, 30°) e al tempo k in (6, 4, 45°), le scansioni in frame locale sono entrambe centrate nell'origine del rispettivo frame, permettendo all'ICP di trovare la trasformazione che meglio allinea i punti
- Trasformazioni per passare tra frame:
  - **World → Local**: Ruota di `-(theta + angle_offset)`, poi trasla di `-(x, y)`
  - **Local → World**: Trasla di `(x, y)`, poi ruota di `(theta + angle_offset)`

**Regola pratica**:
- Visualizzazione grafici → `frame='world'`
- Algoritmo ICP → `frame='local'`
- Analisi geometrica ostacoli → dipende dal contesto

### 5. SIMULAZIONE
- Robot: modello cinematico unicycle
- Ambiente: ostacoli geometrici (Shapely)
- LiDAR: ray-casting con Shapely
- Ground truth: traiettoria perfetta del simulatore

---

## RISULTATI ATTESI

1. **Traiettorie ground truth** precise per ogni movimento
2. **Scansioni LiDAR** realistiche dell'ambiente
3. **Confronto odometria vs ICP**:
   - ICP corregge drift odometrico
   - Inizializzazione migliora convergenza
   - Errore ridotto rispetto a odometria pura
4. **Visualizzazioni**:
   - Plot statici per documentazione
   - Viewer interattivo per esplorazione
   - Grafici di convergenza e analisi

---

## POSSIBILI ESTENSIONI FUTURE

1. Rumore realistico su odometria (slittamento ruote)
2. Ambienti 3D con LiDAR 3D
3. SLAM (Simultaneous Localization And Mapping)
4. Filtro di Kalman per fusione sensori
5. Controllo in closed-loop (path following)
6. Ostacoli dinamici
7. Multi-robot cooperation

---

## NOTE TECNICHE

- **Tempo di simulazione**: Configurabile (default 20-50s)
- **Step temporale**: 0.1s (10 Hz)
- **Risoluzione LiDAR**: 360 raggi
- **Portata LiDAR**: 6 metri
- **Convergenza ICP**: Tipicamente 10-30 iterazioni
- **Tolleranza ICP**: 1e-6 (RMSE)
- **Performance**: Cache ICP per viewer veloce

---

## COME ESEGUIRE

```bash
# Esecuzione standard (configurazione completa automatica)
python src/main.py
# → Salva tutte le immagini (traiettorie, scans LiDAR, analisi ICP)
# → Esegue analisi ICP completa
# → Apre viewer interattivo animato (modalità carousel)

# Solo salvataggio immagini, senza viewer
python src/main.py --skip-viewer

# Solo salvataggio immagini, senza ICP né viewer
python src/main.py --skip-icp --skip-viewer

# Personalizza intervallo di salvataggio scansioni LiDAR
python src/main.py --scan-interval 2.0

# Configurazione LiDAR personalizzata (più raggi, maggiore portata)
python src/main.py --lidar-rays 720 --lidar-range 10.0

# Viewer in modalità grid (5 pannelli ICP) invece di carousel
python src/main.py --viewer-mode grid
```

**Nota**: Quando eseguito senza argomenti, `python src/main.py` attiva automaticamente tutte le funzionalità principali (salvataggio completo + analisi ICP + viewer interattivo).

---

## CONCLUSIONI

Questo progetto dimostra:
- **Integrazione di sensori**: LiDAR per percezione ambiente
- **Localizzazione multi-approccio**: Odometria (interoceptive) vs ICP (exteroceptive)
- **Robustezza**: ICP corregge errori odometrici
- **Simulazione realistica**: Cinematica, sensori, geometria
- **Visualizzazione efficace**: Analisi interattiva e statica

È una base solida per comprendere i fondamenti della robotica mobile, la localizzazione e la percezione dell'ambiente.

