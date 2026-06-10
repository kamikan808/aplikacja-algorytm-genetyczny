import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import pandas as pd
import io

# Importy z Twojego lokalnego pakietu
from ga import (
    GAConfig, run_ga,
    himmelblau, sphere, rastrigin, rosenbrock,
    save_result # Zostawiamy import, gdyby był kiedyś potrzebny
)

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Algorytm Genetyczny", layout="wide")
st.title("Optymalizacja funkcji - Algorytm Genetyczny")

# inicjalizacja historii dla pojedynczych uruchomień
if "history_rows" not in st.session_state:
    st.session_state.history_rows = []

# --- PANEL BOCZNY (PARAMETRY) ---
st.sidebar.header("Konfiguracja Algorytmu")

# Wybór funkcji
funkcja_nazwa = st.sidebar.selectbox(
    "Wybierz funkcję testową", 
    ["Himmelblau", "Sphere", "Rastrigin", "Rosenbrock"]
)

# Słownik do mapowania wyboru na konkretną funkcję z folderu ga
funkcje_map = {
    "Himmelblau": himmelblau,
    "Sphere": sphere,
    "Rastrigin": rastrigin,
    "Rosenbrock": rosenbrock
}
wybrana_funkcja = funkcje_map[funkcja_nazwa]

# Możliwość wyboru liczby zmiennych
liczba_zmiennych = st.sidebar.selectbox("Liczba zmiennych", [2, 5, 10, 20, 27])

# Parametry populacji i epok
pop_size = st.sidebar.number_input("Rozmiar populacji", min_value=10, max_value=1000, value=100, step=10)
epochs = st.sidebar.number_input("Liczba epok", min_value=10, max_value=5000, value=200, step=50)

# Operatory genetyczne
metoda_selekcji = st.sidebar.selectbox("Metoda selekcji", ["Najlepszych", "Ruletki", "Turniejowa"])
metoda_krzyzowania = st.sidebar.selectbox("Krzyżowanie", ["Jednopunktowe", "Dwupunktowe", "Jednorodne", "Ziarniste"])
metoda_mutacji = st.sidebar.selectbox("Mutacja", ["Brzegowa", "Jednopunktowa", "Dwupunktowa"])

# Prawdopodobieństwa
p_crossover = st.sidebar.slider("Prawdopodobieństwo krzyżowania", 0.0, 1.0, 0.8)
p_mutation = st.sidebar.slider("Prawdopodobieństwo mutacji", 0.0, 1.0, 0.1)

# Dodatkowe opcje
inwersja = st.sidebar.checkbox("Zastosuj operator inwersji", value=False)
elitaryzm = st.sidebar.checkbox("Strategia elitarna", value=True)


# --- GŁÓWNA SEKCJA (ZAKŁADKI) ---
tab1, tab2 = st.tabs(["🚀 Pojedyncze uruchomienie", "🏆 Pełny Ranking Metod (Benchmark)"])

# Słowniki mapujące (potrzebne w obu zakładkach)
sel_map = {"Najlepszych": "best", "Ruletki": "roulette", "Turniejowa": "tournament"}
cross_map = {"Jednopunktowe": "single_point", "Dwupunktowe": "two_point", "Jednorodne": "uniform", "Ziarniste": "granular"}
mut_map = {"Brzegowa": "boundary", "Jednopunktowa": "single_point", "Dwupunktowa": "two_point"}

# ==========================================
# ZAKŁADKA 1: POJEDYNCZE URUCHOMIENIE
# ==========================================
with tab1:
    if st.button("Uruchom algorytm", type="primary"):
        
        # Utworzenie konfiguracji 
        cfg = GAConfig(
            func=wybrana_funkcja,
            n_vars=liczba_zmiennych,
            lower=-5.0,
            upper=5.0,
            pop_size=pop_size,
            n_epochs=epochs,
            selection_method=sel_map[metoda_selekcji], 
            crossover_method=cross_map[metoda_krzyzowania],
            mutation_method=mut_map[metoda_mutacji],
            crossover_prob=p_crossover,
            mutation_prob=p_mutation,
            use_inversion=inwersja,
            use_elitism=elitaryzm
        )
        
        # Wykonanie algorytmu z pomiarem czasu
        start_time = time.time()
        
        with st.spinner("Trwają obliczenia..."):
            r = run_ga(cfg)
            
        end_time = time.time()
        czas_obliczen = end_time - start_time
        
        # Wyświetlenie wyników
        st.success("Obliczenia zakończone!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Czas obliczeń", f"{czas_obliczen:.4f} s")
        col2.metric("Najlepsza wartość f(x)", f"{r.best_value:.6f}")
        col3.metric("Liczba epok", epochs)
        
        st.write("**Najlepszy osobnik (X):**")
        st.code(r.best_individual)
        
        # --- DODANIE WYNIKÓW DO PAMIĘCI SESJI ---
        # Zachowujemy pełną zgodność z oryginalnym plikiem results.csv
        nowy_wynik = {
            "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
            "func_name":        funkcja_nazwa,
            "n_vars":           cfg.n_vars,
            "maximize":         cfg.maximize,
            "lower":            cfg.lower,
            "upper":            cfg.upper,
            "pop_size":         cfg.pop_size,
            "n_epochs":         cfg.n_epochs,
            "precision":        cfg.precision,
            "selection_method": cfg.selection_method,
            "crossover_method": cfg.crossover_method,
            "mutation_method":  cfg.mutation_method,
            "crossover_prob":   cfg.crossover_prob,
            "mutation_prob":    cfg.mutation_prob,
            "use_inversion":    cfg.use_inversion,
            "inversion_prob":   cfg.inversion_prob,
            "use_elitism":      cfg.use_elitism,
            "n_elite":          cfg.n_elite,
            "best_value":       round(r.best_value, 10),
            "best_individual":  str([round(v, 6) for v in r.best_individual.tolist()]),
            "elapsed_time":     round(czas_obliczen, 4),
        }
        st.session_state.history_rows.append(nowy_wynik)
        
        # Wykresy 
        if liczba_zmiennych == 2:
            st.subheader("Wizualizacja wyników")
            
            x = np.linspace(-5.0, 5.0, 200)
            y = np.linspace(-5.0, 5.0, 200)
            X, Y = np.meshgrid(x, y)
            
            # Dynamiczne obliczanie Z
            if funkcja_nazwa == "Himmelblau":
                Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2
            elif funkcja_nazwa == "Sphere":
                Z = X**2 + Y**2
            elif funkcja_nazwa == "Rastrigin":
                Z = (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20
            elif funkcja_nazwa == "Rosenbrock":
                Z = 100 * (Y - X**2)**2 + (1 - X)**2
            
            fig = plt.figure(figsize=(14, 5), dpi=100)
            
            # Wykres 3D
            ax2 = fig.add_subplot(1, 2, 1, projection='3d')
            surf = ax2.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='none', alpha=0.8)
            ax2.set_title(f'Wykres 3D - {funkcja_nazwa}')
            fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10, pad=0.1)
            
            # Heatmapa
            ax3 = fig.add_subplot(1, 2, 2)
            heatmap = ax3.contourf(X, Y, Z, levels=50, cmap=cm.inferno)
            ax3.set_title(f'Heatmapa - {funkcja_nazwa}')
            fig.colorbar(heatmap, ax=ax3)
            
            best_x, best_y = r.best_individual[0], r.best_individual[1]
            ax3.plot(best_x, best_y, marker='*', color='cyan', markersize=15, label='Znalezione minimum')
            ax3.legend()
            
            plt.tight_layout()
            st.pyplot(fig)


    # --- TABELA WYNIKÓW I POBIERANIE CSV --- (Pojawia się pod wykresami w zakładce 1)
    if st.session_state.history_rows:
        st.markdown("---")
        st.subheader("📊 Historia uruchomień w tej sesji")
        
        # Konwersja słowników do DataFrame
        df_history = pd.DataFrame(st.session_state.history_rows)
        st.dataframe(df_history, use_container_width=True)
        
        # Zrzut danych do CSV w pamięci RAM
        csv_buffer = io.StringIO()
        df_history.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        # Przycisk pobierania
        st.download_button(
            label="📥 Pobierz całą historię jako CSV (pojedyncze uruchomienia)",
            data=csv_data,
            file_name="wyniki_ga.csv",
            mime="text/csv"
        )


# ==========================================
# ZAKŁADKA 2: RANKING (BENCHMARK)
# ==========================================
with tab2:
    st.write("Ta opcja przetestuje automatycznie **wszystkie 36 kombinacji** (selekcja × krzyżowanie × mutacja) dla aktualnie wybranej funkcji i parametrów (populacja, epoki), a następnie wygeneruje ranking wyników.")
    
    if st.button("Generuj Ranking (Może potrwać kilkadziesiąt sekund)", type="primary", key="btn_ranking"):
        
        lista_selekcji = list(sel_map.keys())
        lista_krzyzowan = list(cross_map.keys())
        lista_mutacji = list(mut_map.keys())
        
        total_runs = len(lista_selekcji) * len(lista_krzyzowan) * len(lista_mutacji)
        current_run = 0
        
        ranking_data = []
        
        # Pasek postępu
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_bench = time.time()
        
        # Trzy zagnieżdżone pętle dla każdej kombinacji
        for sel in lista_selekcji:
            for cross in lista_krzyzowan:
                for mut in lista_mutacji:
                    
                    status_text.text(f"Obliczanie: {sel} | {cross} | {mut} ({current_run+1}/{total_runs})")
                    
                    cfg_bench = GAConfig(
                        func=wybrana_funkcja,
                        n_vars=liczba_zmiennych,
                        lower=-5.0, upper=5.0,
                        pop_size=pop_size, n_epochs=epochs,
                        selection_method=sel_map[sel], 
                        crossover_method=cross_map[cross],
                        mutation_method=mut_map[mut],
                        crossover_prob=p_crossover, mutation_prob=p_mutation,
                        use_inversion=inwersja, use_elitism=elitaryzm
                    )
                    
                    # Czas pojedynczego obiegu
                    t0 = time.time()
                    r_bench = run_ga(cfg_bench)
                    t1 = time.time()
                    
                    ranking_data.append({
                        "Selekcja": sel,
                        "Krzyżowanie": cross,
                        "Mutacja": mut,
                        "Najlepsza wartość": round(r_bench.best_value, 8),
                        "Czas obliczeń [s]": round(t1 - t0, 4)
                    })
                    
                    current_run += 1
                    progress_bar.progress(current_run / total_runs)
        
        status_text.text(f"Zakończono! Całkowity czas benchmarku: {time.time() - start_bench:.2f} s")
        
        # Konwersja do Pandas i sortowanie
        df_ranking = pd.DataFrame(ranking_data)
        
        # Sortowanie po najlepszej wartości
        df_ranking = df_ranking.sort_values(by="Najlepsza wartość", ascending=True).reset_index(drop=True)
        
        # Dodanie kolumny z "Miejscem" w rankingu
        df_ranking.index = df_ranking.index + 1
        df_ranking.index.name = "Pozycja"
        
        st.subheader("🏆 Wyniki Rankingu")
        st.dataframe(df_ranking, use_container_width=True)
        
        # Generowanie pliku CSV z rankingiem
        csv_buffer_rank = io.StringIO()
        df_ranking.to_csv(csv_buffer_rank)
        
        st.download_button(
            label="📥 Pobierz Ranking jako CSV (ranking_summary.csv)",
            data=csv_buffer_rank.getvalue(),
            file_name=f"ranking_summary_{funkcja_nazwa}_{liczba_zmiennych}D.csv",
            mime="text/csv",
            key="download_ranking"
        )