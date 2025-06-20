/* tslint:disable */
/* eslint-disable */
export class EnhancedNWRenderer {
  free(): void;
  constructor();
  render_demo_cube(width: number, height: number, time: number): Promise<any>;
  render_complex_scene(width: number, height: number, time: number): Promise<any>;
  get_device_limits(): object;
}
export class PerformanceMetrics {
  free(): void;
  constructor(width: number, height: number, triangles: number, vertices: number);
  theoretical_max_resolution_for_workgroups(): number;
  readonly triangle_count: number;
  readonly vertex_count: number;
  readonly pixel_count: number;
  readonly workgroup_count: number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_enhancednwrenderer_free: (a: number, b: number) => void;
  readonly enhancednwrenderer_new: () => any;
  readonly enhancednwrenderer_render_demo_cube: (a: number, b: number, c: number, d: number) => any;
  readonly enhancednwrenderer_render_complex_scene: (a: number, b: number, c: number, d: number) => any;
  readonly enhancednwrenderer_get_device_limits: (a: number) => any;
  readonly __wbg_performancemetrics_free: (a: number, b: number) => void;
  readonly performancemetrics_new: (a: number, b: number, c: number, d: number) => number;
  readonly performancemetrics_triangle_count: (a: number) => number;
  readonly performancemetrics_vertex_count: (a: number) => number;
  readonly performancemetrics_pixel_count: (a: number) => number;
  readonly performancemetrics_workgroup_count: (a: number) => number;
  readonly performancemetrics_theoretical_max_resolution_for_workgroups: (a: number) => number;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_export_6: WebAssembly.Table;
  readonly closure58_externref_shim: (a: number, b: number, c: any) => void;
  readonly closure82_externref_shim: (a: number, b: number, c: any, d: any) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
