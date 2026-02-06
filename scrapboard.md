# FILES READ THROUGH

- `./REFACTOR_SUGGESTIONS.md`

## Rest of my solns

tldr:
me figuring out what exactly needs to get done

## Refactors:

### `im2col`: non-naive method (FFT? way that dude did it in numpy? winograd algo?)

### `TensorOps`: every tensor op impl twice: once as `RawTensor`, once as `TensorOps` trait wrapper, remove parallel APIs

delete, test, verify works even without TensorOps

### 400~+ clones: look for hotspots, the ones mentioned might not be the best examples

profile, audit, use refs where possible

### Recursive move op impl

deep nested recursive helpers:
large cyclomatic complexity
param explosion
duplicate code between frob and unfrob
recursive funcs prefent inlining
hard to test individually

#### problems:

could halve file size
lower func call overhead in loops
allow stack traces again

#### to improve:

work on iterator based design (extract to shared iter), removes recursion
cursor/pos struct (look into, see what mean)
generic `transform_indices` helper
look at `ndarray`'s strides pattern

## Improvements:

### `GradFn` macro: every op has custom struct w repetitive boilerplate:

backward impl, clone_box trait method, parent grad extraction (manual), path branching on dev, dev/shape handling
led to repetitive code, error prone (dev handling in new ops), could be done with derive macros (need to research more)
each needs seperate grad check
adding ops needs boilerplate (ew)
manual backwards passes can be error prone

#### todo: `#[derive(Autograd)]` for elem-wise ops

macro system, migrate ops
declarative ops: look at JAX, ATen
auto-gen backward pass from FP and derivative rules
fix dev branching: single loc for CPU/GPU logic

## Testing:

## CI/CD:

## Repo Hygiene:

## GPU:

## Distributed Training:

## Parallelism:

## SIMD:

## ML features:

## Optimizers:

## LR schedulers:

## Kernels:

## Layers:

## Activation Functions:

## Architectures:

## Cargo:

## Package

## Semver

version: `0.3.0`
before `1.0.0` so APIs are unstable
users do not expect things to stick

## Linting

TLDR: yes, i grabbed pretty much everything from clippy
yes, I am going to go through each of these and try to make:
a) the best clippy default config in general
b) the best clippy config for specifically Volta
c) the best clippy defaults for starting new projects
the more tools that statically prove things right or wrong w/o dealing w probabilistic crap improves the performance of coding agents on such a task
bacon, subset of this, good tool that makes working w clippy that much easier

### Clippy

#### Cargo: unadded, only in `bacon cargo` job

##### `cargo_common_metadata`

##### `multiple_crate_versions`

##### `negative_feature_names`

##### `redundant_feature_names`

##### `wildcard_dependencies`

#### Complexity: in `bacon complexity` job

##### `bind_instead_of_map`

##### `bool_comparison`

##### `borrow_deref_ref`

##### `borrowed_box`

##### `bytes_count_to_len`

##### `char_lit_as_u8`

##### `clone_on_copy`

##### `default_constructed_unit_structs`

##### `deprecated_cfg_attr`

##### `deref_addrof`

##### `derivable_impls`

##### `diverging_sub_expression`

##### `double_comparisons`

##### `double_parens`

##### `duration_subsec`

##### `excessive_nesting`

##### `explicit_auto_deref`

##### `explicit_counter_loop`

##### `explicit_write`

##### `extra_unused_lifetimes`

##### `extra_unused_type_parameters`

##### `filter_map_identity`

##### `filter_next`

##### `flat_map_identity`

##### `get_last_with_len`

##### `identity_op`

##### `implied_bounds_in_impls`

##### `inspect_for_each`

##### `int_plus_one`

##### `iter_count`

##### `iter_kv_map`

##### `let_with_type_underscore`

##### `manual_abs_diff`

##### `manual_c_str_literals`

##### `manual_checked_ops`

##### `manual_clamp`

##### `manual_div_ceil`

##### `manual_filter`

##### `manual_filter_map`

##### `manual_find`

##### `manual_find_map`

##### `manual_flatten`

##### `manual_hash_one`

##### `manual_inspect`

##### `manual_is_multiple_of`

##### `manual_main_separator_str`

##### `manual_ok_err`

##### `manual_option_as_slice`

##### `manual_range_patterns`

##### `manual_rem_euclid`

##### `manual_slice_size_calculation`

##### `manual_split_once`

##### `manual_strip`

##### `manual_swap`

##### `manual_take`

##### `manual_unwrap_or`

##### `map_all_any_identity`

##### `map_flatten`

##### `map_identity`

##### `match_as_ref`

##### `match_single_binding`

##### `needless_arbitrary_self_type`

##### `needless_as_bytes`

##### `needless_bool`

##### `needless_bool_assign`

##### `needless_borrowed_reference`

##### `needless_ifs`

##### `needless_lifetimes`

##### `needless_match`

##### `needless_option_as_deref`

##### `needless_option_take`

##### `needless_question_mark`

##### `needless_splitn`

##### `needless_update`

##### `neg_cmp_op_on_partial_ord`

##### `no_effect`

##### `nonminimal_bool`

##### `only_used_in_recursion`

##### `option_as_ref_deref`

##### `option_filter_map`

##### `option_map_unit_fn`

##### `or_then_unwrap`

##### `partialeq_ne_impl`

##### `precedence`

##### `ptr_offset_with_cast`

##### `range_zip_with_len`

##### `redundant_as_str`

##### `redundant_async_block`

##### `redundant_at_rest_pattern`

##### `redundant_closure_call`

##### `redundant_guards`

##### `redundant_slicing`

##### `repeat_once`

##### `reserve_after_initialization`

##### `result_filter_map`

##### `result_map_unit_fn`

##### `seek_from_current`

##### `seek_to_start_instead_of_rewind`

##### `short_circuit_statement`

##### `single_element_loop`

##### `skip_while_next`

##### `string_from_utf8_as_bytes`

##### `strlen_on_c_strings`

##### `swap_with_temporary`

##### `temporary_assignment`

##### `too_many_arguments`

##### `transmute_bytes_to_str`

##### `transmute_int_to_bool`

##### `transmute_int_to_non_zero`

##### `transmute_ptr_to_ref`

##### `transmutes_expressible_as_ptr_casts`

##### `type_complexity`

##### `unit_arg`

##### `unnecessary_cast`

##### `unnecessary_filter_map`

##### `unnecessary_find_map`

##### `unnecessary_first_then_check`

##### `unnecessary_literal_unwrap`

##### `unnecessary_map_on_constructor`

##### `unnecessary_min_or_max`

##### `unnecessary_operation`

##### `unnecessary_sort_by`

##### `unnecessary_unwrap`

##### `unneeded_wildcard_pattern`

##### `unused_format_specs`

##### `useless_asref`

##### `useless_concat`

##### `useless_conversion`

##### `useless_format`

##### `useless_nonzero_new_unchecked`

##### `useless_transmute`

##### `vec_box`

##### `while_let_loop`

##### `wildcard_in_or_patterns`

##### `zero_divided_by_zero`

##### `zero_prefixed_literal`

#### Correctness: not sure

##### `absurd_extreme_comparisons`

##### `almost_swapped`

##### `approx_constant`

##### `async_yields_async`

##### `bad_bit_mask`

##### `cast_slice_different_sizes`

##### `char_indices_as_byte_indices`

##### `deprecated_semver`

##### `derive_ord_xor_partial_ord`

##### `derived_hash_with_manual_eq`

##### `eager_transmute`

##### `enum_clike_unportable_variant`

##### `eq_op`

##### `erasing_op`

##### `if_let_mutex`

##### `ifs_same_cond`

##### `impl_hash_borrow_with_str_and_bytes`

##### `impossible_comparisons`

##### `ineffective_bit_mask`

##### `infinite_iter`

##### `inherent_to_string_shadow_display`

##### `inline_fn_without_body`

##### `invalid_regex`

##### `inverted_saturating_sub`

##### `invisible_characters`

##### `iter_next_loop`

##### `iter_skip_zero`

##### `iterator_step_by_zero`

##### `let_underscore_lock`

##### `lint_groups_priority`

##### `match_str_case_mismatch`

##### `mem_replace_with_uninit`

##### `min_max`

##### `mistyped_literal_suffixes`

##### `modulo_one`

##### `mut_from_ref`

##### `never_loop`

##### `non_octal_unix_permissions`

##### `nonsensical_open_options`

##### `not_unsafe_ptr_arg_deref`

##### `option_env_unwrap`

##### `out_of_bounds_indexing`

##### `overly_complex_bool_expr`

##### `panicking_overflow_checks`

##### `panicking_unwrap`

##### `possible_missing_comma`

##### `read_line_without_trim`

##### `recursive_format_impl`

##### `redundant_comparisons`

##### `reversed_empty_ranges`

##### `self_assignment`

##### `serde_api_misuse`

##### `size_of_in_element_count`

##### `suspicious_splitn`

##### `transmute_null_to_fn`

##### `transmuting_null`

##### `uninit_assumed_init`

##### `uninit_vec`

##### `unit_cmp`

##### `unit_hash`

##### `unit_return_expecting_ord`

##### `unsound_collection_transmute`

##### `unused_io_amount`

##### `useless_attribute`

##### `vec_resize_to_zero`

##### `while_immutable_condition`

##### `wrong_transmute`

##### `zst_offset`

#### Nursery: got a lot of these earlier but not sure

##### `as_ptr_cast_mut`

##### `branches_sharing_code`

##### `clear_with_drain`

##### `coerce_container_to_any`

##### `collection_is_never_read`

##### `debug_assert_with_mut_call`

##### `derive_partial_eq_without_eq`

##### `doc_link_code`

##### `equatable_if_let`

##### `fallible_impl_from`

##### `future_not_send`

##### `imprecise_flops`

##### `iter_on_empty_collections`

##### `iter_on_single_items`

##### `iter_with_drain`

##### `large_stack_frames`

##### `literal_string_with_formatting_args`

##### `missing_const_for_fn`

##### `needless_collect`

##### `needless_pass_by_ref_mut`

##### `needless_type_cast`

##### `non_send_fields_in_send_ty`

##### `nonstandard_macro_braces`

##### `option_if_let_else`

##### `or_fun_call`

##### `path_buf_push_overwrite`

##### `read_zero_byte_vec`

##### `redundant_clone`

##### `redundant_pub_crate`

##### `search_is_some`

##### `set_contains_or_insert`

##### `significant_drop_in_scrutinee`

##### `significant_drop_tightening`

##### `single_option_map`

##### `string_lit_as_bytes`

##### `suboptimal_flops`

##### `suspicious_operation_groupings`

##### `too_long_first_doc_paragraph`

##### `trailing_empty_array`

##### `trait_duplication_in_bounds`

##### `transmute_undefined_repr`

##### `trivial_regex`

##### `tuple_array_conversions`

##### `type_repetition_in_bounds`

##### `uninhabited_references`

##### `unnecessary_struct_initialization`

##### `unused_peekable`

##### `unused_rounding`

##### `use_self`

##### `useless_let_if_seq`

##### `volatile_composites`

##### `while_float`

#### Pedantic: actually had a decent number of these from bacon job

##### `assigning_clones`

##### `bool_to_int_with_if`

##### `borrow_as_ptr`

##### `case_sensitive_file_extension_comparisons`

##### `cast_lossless`

##### `cast_possible_truncation`

##### `cast_possible_wrap`

##### `cast_precision_loss`

##### `cast_ptr_alignment`

##### `cast_sign_loss`

##### `checked_conversions`

##### `cloned_instead_of_copied`

##### `collapsible_else_if`

##### `comparison_chain`

##### `copy_iterator`

##### `decimal_bitwise_operands`

##### `default_trait_access`

##### `doc_broken_link`

##### `doc_comment_double_space_linebreaks`

##### `doc_link_with_quotes`

##### `doc_markdown`

##### `duration_suboptimal_units`

##### `elidable_lifetime_names`

##### `empty_enums`

##### `enum_glob_use`

##### `expl_impl_clone_on_copy`

##### `explicit_deref_methods`

##### `explicit_into_iter_loop`

##### `explicit_iter_loop`

##### `filter_map_next`

##### `flat_map_option`

##### `float_cmp`

##### `fn_params_excessive_bools`

##### `format_collect`

##### `format_push_string`

##### `from_iter_instead_of_collect`

##### `if_not_else`

##### `ignore_without_reason`

##### `ignored_unit_patterns`

##### `implicit_clone`

##### `implicit_hasher`

##### `inconsistent_struct_constructor`

##### `index_refutable_slice`

##### `inefficient_to_string`

##### `inline_always`

##### `into_iter_without_iter`

##### `invalid_upcast_comparisons`

##### `ip_constant`

##### `items_after_statements`

##### `iter_filter_is_ok`

##### `iter_filter_is_some`

##### `iter_not_returning_iterator`

##### `iter_without_into_iter`

##### `large_digit_groups`

##### `large_futures`

##### `large_stack_arrays`

##### `large_types_passed_by_value`

##### `linkedlist`

##### `macro_use_imports`

##### `manual_assert`

##### `manual_ilog2`

##### `manual_instant_elapsed`

##### `manual_is_power_of_two`

##### `manual_is_variant_and`

##### `manual_let_else`

##### `manual_midpoint`

##### `manual_string_new`

##### `many_single_char_names`

##### `map_unwrap_or`

##### `match_bool`

##### `match_same_arms`

##### `match_wild_err_arm`

##### `match_wildcard_for_single_variants`

##### `maybe_infinite_iter`

##### `mismatching_type_param_order`

##### `missing_errors_doc`

##### `missing_fields_in_debug`

##### `missing_panics_doc`

##### `must_use_candidate`

##### `mut_mut`

##### `naive_bytecount`

##### `needless_bitwise_bool`

##### `needless_continue`

##### `needless_for_each`

##### `needless_pass_by_value`

##### `needless_raw_string_hashes`

##### `no_effect_underscore_binding`

##### `no_mangle_with_rust_abi`

##### `non_std_lazy_statics`

##### `option_as_ref_cloned`

##### `option_option`

##### `ptr_as_ptr`

##### `ptr_cast_constness`

##### `ptr_offset_by_literal`

##### `pub_underscore_fields`

##### `range_minus_one`

##### `range_plus_one`

##### `redundant_closure_for_method_calls`

##### `redundant_else`

##### `ref_as_ptr`

##### `ref_binding_to_reference`

##### `ref_option`

##### `ref_option_ref`

##### `return_self_not_must_use`

##### `same_functions_in_if_condition`

##### `same_length_and_capacity`

##### `self_only_used_in_recursion`

##### `semicolon_if_nothing_returned`

##### `should_panic_without_expect`

##### `similar_names`

##### `single_char_pattern`

##### `single_match_else`

##### `stable_sort_primitive`

##### `str_split_at_newline`

##### `string_add_assign`

##### `struct_excessive_bools`

##### `struct_field_names`

##### `too_many_lines`

##### `transmute_ptr_to_ptr`

##### `trivially_copy_pass_by_ref`

##### `unchecked_time_subtraction`

##### `unicode_not_nfc`

##### `uninlined_format_args`

##### `unnecessary_box_returns`

##### `unnecessary_debug_formatting`

##### `unnecessary_join`

##### `unnecessary_literal_bound`

##### `unnecessary_semicolon`

##### `unnecessary_wraps`

##### `unnested_or_patterns`

##### `unreadable_literal`

##### `unsafe_derive_deserialize`

##### `unused_async`

##### `unused_self`

##### `used_underscore_binding`

##### `used_underscore_items`

##### `verbose_bit_mask`

##### `wildcard_imports`

##### `zero_sized_map_values`

#### Perf: also not sure if i did any of these

##### `box_collection`

##### `boxed_local`

##### `cloned_ref_to_slice_refs`

##### `cmp_owned`

##### `collapsible_str_replace`

##### `double_ended_iterator_last`

##### `drain_collect`

##### `expect_fun_call`

##### `extend_with_drain`

##### `format_in_format_args`

##### `iter_overeager_cloned`

##### `large_const_arrays`

##### `large_enum_variant`

##### `manual_contains`

##### `manual_ignore_case_cmp`

##### `manual_memcpy`

##### `manual_retain`

##### `manual_str_repeat`

##### `manual_try_fold`

##### `map_entry`

##### `missing_const_for_thread_local`

##### `missing_spin_loop`

##### `readonly_write_lock`

##### `redundant_allocation`

##### `redundant_iter_cloned`

##### `regex_creation_in_loops`

##### `replace_box`

##### `result_large_err`

##### `sliced_string_as_bytes`

##### `slow_vector_initialization`

##### `to_string_in_format_args`

##### `unbuffered_bytes`

##### `unnecessary_to_owned`

##### `useless_vec`

##### `vec_init_then_push`

##### `waker_clone_wake`

#### Style: idk

##### `assertions_on_constants`

##### `assign_op_pattern`

##### `blocks_in_conditions`

##### `bool_assert_comparison`

##### `borrow_interior_mutable_const`

##### `box_default`

##### `builtin_type_shadow`

##### `byte_char_slices`

##### `bytes_nth`

##### `chars_last_cmp`

##### `chars_next_cmp`

##### `cmp_null`

##### `collapsible_if`

##### `collapsible_match`

##### `comparison_to_empty`

##### `default_instead_of_iter_empty`

##### `disallowed_macros`

##### `disallowed_methods`

##### `disallowed_names`

##### `disallowed_types`

##### `doc_lazy_continuation`

##### `doc_overindented_list_items`

##### `double_must_use`

##### `duplicate_underscore_argument`

##### `enum_variant_names`

##### `err_expect`

##### `excessive_precision`

##### `field_reassign_with_default`

##### `filter_map_bool_then`

##### `fn_to_numeric_cast`

##### `fn_to_numeric_cast_with_truncation`

##### `for_kv_map`

##### `from_over_into`

##### `from_str_radix_10`

##### `get_first`

##### `if_same_then_else`

##### `implicit_saturating_add`

##### `implicit_saturating_sub`

##### `inconsistent_digit_grouping`

##### `infallible_destructuring_match`

##### `inherent_to_string`

##### `init_numbered_fields`

##### `into_iter_on_ref`

##### `io_other_error`

##### `is_digit_ascii_radix`

##### `items_after_test_module`

##### `iter_cloned_collect`

##### `iter_next_slice`

##### `iter_nth`

##### `iter_nth_zero`

##### `iter_skip_next`

##### `just_underscores_and_digits`

##### `legacy_numeric_constants`

##### `len_without_is_empty`

##### `len_zero`

##### `let_and_return`

##### `let_unit_value`

##### `main_recursion`

##### `manual_async_fn`

##### `manual_bits`

##### `manual_dangling_ptr`

##### `manual_is_ascii_check`

##### `manual_is_finite`

##### `manual_is_infinite`

##### `manual_map`

##### `manual_next_back`

##### `manual_non_exhaustive`

##### `manual_ok_or`

##### `manual_pattern_char_comparison`

##### `manual_range_contains`

##### `manual_repeat_n`

##### `manual_rotate`

##### `manual_saturating_arithmetic`

##### `manual_slice_fill`

##### `manual_while_let_some`

##### `map_clone`

##### `map_collect_result_unit`

##### `match_like_matches_macro`

##### `match_overlapping_arm`

##### `match_ref_pats`

##### `match_result_ok`

##### `mem_replace_option_with_none`

##### `mem_replace_option_with_some`

##### `mem_replace_with_default`

##### `missing_enforced_import_renames`

##### `missing_safety_doc`

##### `mixed_attributes_style`

##### `mixed_case_hex_literals`

##### `module_inception`

##### `multiple_bound_locations`

##### `must_use_unit`

##### `mut_mutex_lock`

##### `needless_borrow`

##### `needless_borrows_for_generic_args`

##### `needless_doctest_main`

##### `needless_else`

##### `needless_late_init`

##### `needless_parens_on_range_literals`

##### `needless_pub_self`

##### `needless_range_loop`

##### `needless_return`

##### `needless_return_with_question_mark`

##### `neg_multiply`

##### `new_ret_no_self`

##### `new_without_default`

##### `non_minimal_cfg`

##### `obfuscated_if_else`

##### `ok_expect`

##### `op_ref`

##### `option_map_or_none`

##### `owned_cow`

##### `partialeq_to_none`

##### `print_literal`

##### `print_with_newline`

##### `println_empty_string`

##### `ptr_arg`

##### `ptr_eq`

##### `question_mark`

##### `redundant_closure`

##### `redundant_field_names`

##### `redundant_pattern`

##### `redundant_pattern_matching`

##### `redundant_static_lifetimes`

##### `result_map_or_into_option`

##### `result_unit_err`

##### `same_item_push`

##### `self_named_constructors`

##### `should_implement_trait`

##### `single_char_add_str`

##### `single_component_path_imports`

##### `single_match`

##### `string_extend_chars`

##### `tabs_in_doc_comments`

##### `to_digit_is_some`

##### `to_string_trait_impl`

##### `toplevel_ref_arg`

##### `trim_split_whitespace`

##### `unnecessary_fallible_conversions`

##### `unnecessary_fold`

##### `unnecessary_lazy_evaluations`

##### `unnecessary_map_or`

##### `unnecessary_mut_passed`

##### `unnecessary_owned_empty_strings`

##### `unneeded_struct_pattern`

##### `unsafe_removed_from_name`

##### `unused_enumerate_index`

##### `unused_unit`

##### `unusual_byte_groupings`

##### `unwrap_or_default`

##### `upper_case_acronyms`

##### `while_let_on_iterator`

##### `write_literal`

##### `write_with_newline`

##### `writeln_empty_string`

##### `wrong_self_convention`

##### `zero_ptr`

#### Suspicious

##### `almost_complete_range`

##### `arc_with_non_send_sync`

##### `await_holding_invalid_type`

##### `await_holding_lock`

##### `await_holding_refcell_ref`

##### `blanket_clippy_restriction_lints`

##### `cast_abs_to_unsigned`

##### `cast_enum_constructor`

##### `cast_enum_truncation`

##### `cast_nan_to_int`

##### `cast_slice_from_raw_parts`

##### `confusing_method_to_numeric_cast`

##### `const_is_empty`

##### `crate_in_macro_def`

##### `crosspointer_transmute`

##### `declare_interior_mutable_const`

##### `deprecated_clippy_cfg_attr`

##### `doc_nested_refdefs`

##### `doc_suspicious_footnotes`

##### `drop_non_drop`

##### `duplicate_mod`

##### `duplicated_attributes`

##### `empty_docs`

##### `empty_line_after_doc_comments`

##### `empty_line_after_outer_attr`

##### `empty_loop`

##### `float_equality_without_abs`

##### `forget_non_drop`

##### `four_forward_slashes`

##### `from_raw_with_void_ptr`

##### `incompatible_msrv`

##### `ineffective_open_options`

##### `infallible_try_from`

##### `iter_out_of_bounds`

##### `join_absolute_paths`

##### `let_underscore_future`

##### `lines_filter_map_ok`

##### `macro_metavars_in_unsafe`

##### `manual_unwrap_or_default`

##### `misnamed_getters`

##### `misrefactored_assign_op`

##### `missing_transmute_annotations`

##### `multi_assignments`

##### `mut_range_bound`

##### `mutable_key_type`

##### `needless_character_iteration`

##### `needless_maybe_sized`

##### `no_effect_replace`

##### `non_canonical_clone_impl`

##### `non_canonical_partial_ord_impl`

##### `octal_escapes`

##### `path_ends_with_ext`

##### `permissions_set_readonly_false`

##### `pointers_in_nomem_asm_block`

##### `possible_missing_else`

##### `print_in_format_impl`

##### `rc_clone_in_vec_init`

##### `redundant_locals`

##### `repeat_vec_with_capacity`

##### `repr_packed_without_abi`

##### `single_range_in_vec_init`

##### `size_of_ref`

##### `suspicious_arithmetic_impl`

##### `suspicious_assignment_formatting`

##### `suspicious_command_arg_space`

##### `suspicious_doc_comments`

##### `suspicious_else_formatting`

##### `suspicious_map`

##### `suspicious_op_assign_impl`

##### `suspicious_open_options`

##### `suspicious_to_owned`

##### `suspicious_unary_op_formatting`

##### `swap_ptr_to_ref`

##### `test_attr_in_doctest`

##### `type_id_on_box`

##### `unconditional_recursion`

##### `unnecessary_clippy_cfg`

##### `unnecessary_get_then_check`

##### `unnecessary_option_map_or_else`

##### `unnecessary_result_map_or_else`

##### `zero_repeat_side_effects`

##### `zombie_processes`
