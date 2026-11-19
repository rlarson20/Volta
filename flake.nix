{
  description = "Volta dev shells: stable Rust plus the tools the justfile calls";

  # `nix develop` (or direnv's `use flake`) enters `default`: CPU work, `just ci`'s tools,
  # `just serve-docs` and `just torch-fixtures`. `.#gpu` (Linux) adds Mesa's lavapipe for
  # `just test-gpu` without a GPU, as the CI lavapipe lane does. `.#msrv` is the 1.89 lane.
  # dev-docs/DEVELOPMENT.md, "Nix dev shell", says what each covers and what it doesn't.

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    fenix.url = "github:nix-community/fenix";
    fenix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { nixpkgs, fenix, ... }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f {
        pkgs = nixpkgs.legacyPackages.${system};
        fx = fenix.packages.${system};
      });

      components = [
        "cargo" "clippy" "rust-src" "rustc" "rustfmt" "rust-analyzer"
        "llvm-tools-preview" # cargo llvm-cov
      ];

      # Must match `rust-version` in Cargo.toml and the `1.89` CI matrix lane.
      msrv = { channel = "1.89.0"; sha256 = "sha256-+9FmLhAOezBZCOziO0Qct1NOrfpjNsXxc/8I0c7BdKE="; };
    in {
      devShells = forAll ({ pkgs, fx }:
        let
          inherit (pkgs) lib stdenv;

          # Stable, pinned by flake.lock (`nix flake update fenix` to move it), with the two
          # cross targets CI checks: `just wasm-check` and `just macos-cross`.
          rust = fx.combine [
            (fx.stable.withComponents components)
            fx.targets.wasm32-unknown-unknown.stable.rust-std
            fx.targets.aarch64-apple-darwin.stable.rust-std
          ];

          tools = with pkgs; [
            # First, so `python3` is 3.12 rather than the Python pre-commit brings along.
            python312
            uv # `just serve-docs`, `just torch-fixtures`
            just
            sccache
            pre-commit
            gh # `just check-ignores` (issue half), `just issues`, `just prs`
            git-cliff # release notes (volta-tracking release plan)
            bacon # bacon.toml
            cargo-hack
            cargo-llvm-cov
            cargo-mutants
            cargo-deny
            cargo-machete
            cargo-expand
            cargo-bloat
          ] ++ lib.optionals stdenv.hostPlatform.isLinux [
            # `just macos-cross` compiles criterion's one C file for aarch64-apple-darwin with
            # clang's freestanding headers. Unwrapped: the cc-wrapper adds Linux glibc flags.
            llvmPackages.clang-unwrapped
          ];

          env = {
            RUSTC_WRAPPER = "sccache";
            # uv uses this Python, never one it downloads (a downloaded one won't run on NixOS).
            UV_PYTHON_DOWNLOADS = "never";
            UV_PYTHON = "${pkgs.python312}/bin/python3.12";
          } // lib.optionalAttrs stdenv.hostPlatform.isLinux {
            # torch's wheel (`just torch-fixtures`) dlopens libstdc++ and libz, which a Nix
            # Python doesn't provide.
            LD_LIBRARY_PATH = lib.makeLibraryPath [ stdenv.cc.cc.lib pkgs.zlib ];
          };

          mkShell = packages: extraEnv: pkgs.mkShell ({ inherit packages; } // env // extraEnv);
        in
        {
          default = mkShell ([ rust ] ++ tools) { };

          # MSRV check only (`cargo +1.89 ...` in CI): build, clippy and tests on 1.89.
          msrv = mkShell [ ((fx.toolchainOf msrv).withComponents [ "cargo" "clippy" "rustc" "rustfmt" ]) pkgs.just pkgs.sccache ] { };
        } // lib.optionalAttrs stdenv.hostPlatform.isLinux {
          # `VOLTA_REQUIRE_GPU=1 just test-gpu` on a machine without a GPU: wgpu dlopens the
          # Vulkan loader, and VK_ADD_DRIVER_FILES adds lavapipe (software Vulkan) next to any
          # driver NixOS already provides in /run/opengl-driver.
          gpu = mkShell ([ rust pkgs.vulkan-tools ] ++ tools) {
            LD_LIBRARY_PATH = lib.makeLibraryPath [ stdenv.cc.cc.lib pkgs.zlib pkgs.vulkan-loader ];
            VK_ADD_DRIVER_FILES = "${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.${stdenv.hostPlatform.uname.processor}.json";
          };
        });
    };
}
