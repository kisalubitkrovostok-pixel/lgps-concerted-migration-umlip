# Материалы для отчёта и презентации


**Проект:** concerted (коллективная) миграция Li с участием вакансии в Li₁₀GeP₂S₁₂ (LGPS):
проверка универсального эквивариантного MLIP (SevenNet).

**Задачи (по замечаниям инструктора):**
1. Проверить, воспроизводит ли uMLIP поверхность потенциальной энергии (PES) для concerted-миграции → **NEB**.
2. Посчитать число concerted-событий в одной траектории при одной температуре и сравнить с литературой → **MD 700 K**.

Один материал (LGPS), одна модель (SevenNet), одна температура (700 K).
Исходные данные скопированы с сервера `fjord2:~/umlip-benchmark` 2026-10-22;
полное зеркало лежит в `../server_umlip_benchmark/`.

---

## 01_figures — картинки для отчёта и слайдов

| Файл | Что показывает | Куда в отчёт |
|---|---|---|
| `fig1_NEB_SevenNet_single_vs_concerted_vs_DFT.png` | NEB-профили для прыжка одного иона и для concerted-миграции 2/3/4 ионов вдоль c-канала; пунктиром DFT-барьеры (He et al., 2017) | Результаты, задача 1 (главная картинка) |
| `fig2_NEB_SevenNet_on_hops_extracted_from_MD.png` | NEB для реальных прыжков, вырезанных из MD-траектории | Результаты, задача 1 (дополнение) |
| `fig3_SevenNet_700K_Li_displacement_and_event_size_distribution.png` | Слева: распределение смещений Li за τ = 2.35 пс и порог прыжка; справа: распределение размеров событий P(n) | Результаты, задача 2 (главная картинка) |
| `fig4_concerted_event_count_sensitivity_to_parameters.png` | Как число concerted-событий зависит от параметров детекции (d_hop, τ, r_cut): 27 комбинаций | Результаты, задача 2 / обсуждение надёжности |
| `figS2_SevenNet_700K_MD_check_full_300ps_MSD_T_Epot.png` | Проверка MD: MSD Li (D* = 2.5·10⁻⁵ см²/с; на одно направление D_c ≈ 2·D_ab), стабильность T и E_pot | Методы / приложение |
| `figS3_event_size_SevenNet_vs_MACE_supplementary.png` | P(n) для SevenNet и MACE-MP-0 | Только приложение (вторая модель вне рамок задания) |

Все картинки, кроме fig3, строит `06_scripts_methods/make_report_figures.py` из данных этой папки.

## Ключевые числа (SevenNet, 700 K, 200 атомов, 80 Li, NVT, шаг 2 fs, 300 пс)

| Величина | Значение | Источник |
|---|---|---|
| Прыжков Li всего | 3790 | `02_.../SevenNet_700K_concerted_summary_main_parameters.json` |
| Событий всего / из них concerted (n ≥ 2) | 1556 / 905 | там же |
| Concerted-событий в пс | 3.0 (от 1.8 до 4.8 при изменении параметров) | там же + `..._parameter_scan_27_combinations.csv` |
| Доля прыжков в concerted-событиях | 83 % (от 50 до 95 % при изменении параметров) | там же |
| Средний размер события (взвешенный по ионам) | 3.9 иона | там же |
| D* (tracer), см²/с | 2.5·10⁻⁵ | там же, проверено по MSD (figS2) |
| NEB, идеальный c-канал: барьер одного иона / 2 / 3 / 4 ионов, эВ | 0.047 / 0.027 / 0.012 / 0.152 | `03_.../A_.../summary_barriers.json` |
| NEB, прыжки из MD (E_KRA), эВ | 0.063, 0.068 | `03_.../B_.../barriers_from_MD_events.csv` |
| DFT (He et al., 2017) | ~0.2 эВ concerted, ~0.5 эВ один ион | литература |
| Стоимость | ~0.5 с/шаг на 8 ядрах, 300 пс ≈ 20 ч | `04_.../info.json` |

Параметры детекции событий по умолчанию: d_hop = 1.8 Å, τ = 2.35 пс, r_cut = 2.0 Å (метод directed graph, Sato et al., 2026).

**Важно для выводов:** все барьеры SevenNet в несколько раз ниже DFT. Это согласуется с известным
смягчением PES у uMLIP, но конечные состояния NEB нужно проверить. В отчёте стоит указать это как
главный результат / ограничение.

---

## 02_data_concerted_events — данные для задачи 2
- `SevenNet_700K_concerted_summary_main_parameters.json` — все итоговые числа при основных параметрах (включая разбивку по размеру события n).
- `SevenNet_700K_event_size_histogram.csv` — N(n) и P(n): таблица для сравнения с литературой.
- `SevenNet_700K_parameter_scan_27_combinations.csv` — результаты для всех 27 комбинаций параметров (данные для fig4).
- `SevenNet_700K_all_events_list.csv` — список всех найденных событий.
- `analysis_parameters_note.txt` — какие параметры перебирались.

## 03_data_NEB — данные для задачи 1
- `A_idealized_c_channel_NEB/` — CI-NEB в суперячейке 2×1×2 (199 атомов, одна вакансия Li); двигаются 1, 2, 3 или 4 иона Li вдоль c-канала. Все 4 расчёта сошлись. Профили `profile_n*.csv`, пути `path_n*.traj` (открываются в ASE / OVITO; из них можно делать картинки пути для слайдов), логи и `summary_barriers.json`.
- `B_NEB_on_events_from_MD/` — NEB для прыжков, вырезанных из MD-траектории. Сошлись только 2 события с одним ионом (cand8, cand11). Для n = 2–4 подходящих изолированных событий не нашлось (см. `run_n*_5540547.out`).

## 04_data_MD_run — сама траектория 700 K
- `info.json`, `settings.json`: параметры прогона (T, шаг, термостат, число кадров, скорость счёта).
- `log.csv`, `equil.csv`: T и E_pot по времени (продуктивная часть / эквилибрация).
- `timestep_test_1fs_vs_2fs.json`: проверка, что шаг 2 fs допустим (дрейф энергии ≈ 0).
- `md_trajectory_300ps_LINK.traj`: **символическая ссылка** на траекторию (64 МБ) в `../server_umlip_benchmark/`.

## 05_structure
- `LGPS.cif`: упорядоченная структура LGPS; `LGPS_source.json`: откуда она взята (Materials Project).

## 06_scripts_methods — код (для раздела «Методы» и воспроизводимости)
- `run_md.py`, `md.sbatch`: NVT MD.
- `concerted_events.py`, `concerted_scan.py`: поиск прыжков и группировка их в concerted-события + перебор параметров.
- `neb_lgps_sevennet.py`: NEB в идеальном c-канале (набор A).
- `neb_from_md.py`, `neb_md.sbatch`: NEB по событиям из MD (набор B).
- `dt_test_sevennet.py`: тест шага по времени.
- `environment-sevennet.txt`: версии пакетов.
- `make_report_figures.py`: строит fig1, fig2, fig4, figS2, figS3.

## 07_supplementary_NOT_main_text — в основной текст НЕ брать
- `MACE_700K_comparison/`: тот же анализ для MACE-MP-0 (175 пс). Результат почти такой же (3.1 concerted-событий в пс, 80 %, D* = 2.5·10⁻⁵). Можно одной фразой упомянуть в обсуждении как проверку на другой модели.
- `NEB_unconverged_DO_NOT_USE/`: NEB с закреплёнными Li, которые не сошлись (барьеры 2–20 эВ, нефизичные). **Не использовать.**
- `other_temperatures_LGPS/`: MD при 900 и 1100 K. По заданию нужна одна температура.

Прогоны для LLZO остались только в `../server_umlip_benchmark/runs/`: по заданию нужен один материал.

## 08_report_LaTeX — итоговый отчёт
- `report.tex` + `figures/` — исходник отчёта (можно загрузить в Overleaf как есть: report.tex и папку figures).
- `report.pdf` — скомпилированный отчёт (копия: `../Final_Project_Report_LGPS_concerted_migration.pdf`).
- Рисунки отчёта (300 dpi) строит `06_scripts_methods/make_paper_figures.py`; он же печатает D*, N_eff и RSD(D), которые цитируются в тексте.
- `02_data_concerted_events/*_Sato_parameters_dhop1.4.*` — анализ событий с параметрами Sato et al. (d_hop = 1.4 Å), посчитан локально для таблицы 2.
