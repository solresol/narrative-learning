CREATE TABLE exoplanets (
  decodex INTEGER,
  exoplanet_id text PRIMARY KEY,
  mean_orbital_radius text,
  mean_surface_roughness text,
  mean_magnetosphere_extent text,
  mean_atmospheric_depth text,
  mean_tidal_distortion text,
  mean_core_density text,
  mean_ring_system_complexity text,
  mean_impact_crater_count text,
  mean_axial_symmetry text,
  mean_cloud_turbulence text,
  orbital_radius_error text,
  surface_roughness_error text,
  magnetosphere_extent_error text,
  atmospheric_depth_error text,
  tidal_distortion_error text,
  core_density_error text,
  ring_system_complexity_error text,
  impact_crater_count_error text,
  axial_symmetry_error text,
  cloud_turbulence_error text,
  max_orbital_radius text,
  max_surface_roughness text,
  max_magnetosphere_extent text,
  max_atmospheric_depth text,
  max_tidal_distortion text,
  max_core_density text,
  max_ring_system_complexity text,
  max_impact_crater_count text,
  max_axial_symmetry text,
  max_cloud_turbulence text,
  occupying_species text
);
CREATE TABLE inferences (
  round_id integer references rounds(round_id),
  creation_time datetime default current_timestamp,
  exoplanet_id text references exoplanets(exoplanet_id),
  narrative_text text,
  llm_stderr text,
  prediction text,
  primary key (round_id, exoplanet_id)
);
CREATE TABLE splits (
  split_id integer primary key autoincrement
);
CREATE TABLE sqlite_sequence(name,seq);
CREATE TABLE exoplanet_splits (
  split_id integer references splits(split_id),
  exoplanet_id text references exoplanets(exoplanet_id),
  holdout bool not null default false, -- holdout either for validation or test
  validation bool not null default false,
  primary key (split_id, exoplanet_id)
);
CREATE TABLE rounds (
  round_id integer primary key autoincrement,
  round_start datetime default current_timestamp,
  split_id integer references splits(split_id),
  prompt text,
  reasoning_for_this_prompt text,
  stderr_from_prompt_creation text
);
