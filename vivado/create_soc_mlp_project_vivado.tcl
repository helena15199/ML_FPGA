# -----------------------------------------------------------------------------
# Auto-generated Vivado project script for Ultra96 sequential AXI design.
# Layout    : mem_layout.json
# Topology  : 6 -> 66 -> 50 -> 38 -> 28 -> 20 -> 2
# Mode      : sequential
# -----------------------------------------------------------------------------

proc print_help {} {
    puts "\nUsage: source [info script] -tclargs ?options?"
    puts "Options:"
    puts "  --run                  Run synthesis and implementation up to bitstream."
    puts "  --project-name <name>  Vivado project name."
    puts "  --project-dir <path>   Vivado project directory."
    puts "  --part <part>          FPGA part."
    puts "  --board-part <board>   Optional board_part. Empty disables it."
    puts "  --top <module>         RTL top for IP packaging. Default: SOC_MLP."
    puts "  --bd-name <name>       Block design name. Default: design_1."
    puts "  --ip-repo-dir <path>   Generated IP repository. Default: <repo>/vivado/ip_repo."
    puts "  --skip-ip              Do not package SOC_MLP as local IP."
    puts "  --skip-bd              Do not create the block design/wrapper."
    puts "  --jobs <N>             Vivado run parallelism."
    puts "  --ip-clk-freq-hz <Hz>  Packaged SOC_MLP AXI clock frequency. Default: 99999001."
    puts "  --ignore-freq-hz       Mark FREQ_HZ compatibility as ignored when supported."
    puts "  --strict-freq-hz       Do not request Ignore FREQ_HZ compatibility."
    puts "  --help                 Print this help.\n"
}

proc add_existing_files {fileset files file_type required} {
    foreach f $files {
        set nf [file normalize $f]
        if {[file exists $nf]} {
            add_files -norecurse -fileset $fileset $nf
            set obj [get_files -of_objects [get_filesets $fileset] [list $nf]]
            if {$obj ne ""} { set_property file_type $file_type $obj }
            puts "INFO: added $nf"
        } else {
            if {$required} { error "Required file not found: $nf" }
            puts "WARNING: optional file not found: $nf"
        }
    }
}

set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file normalize [file join $script_dir ".."]]

set project_name "SOC_MLP_ultra96_sequential"
set top_name     "SOC_MLP"
set part_name    "xczu3eg-sbva484-1-e"
set board_part   ""
set jobs         8
set ip_clk_freq_hz 99999001
set ignore_freq_hz_compat 1
set run_flow     0
set create_ip    1
set create_bd    1
set project_dir  ""
set bd_name      "design_1"
set ip_repo_dir  ""
set ip_vendor    "user.org"
set ip_library   "user"
set ip_name      "SOC_MLP"
set ip_version   "1.0"

for {set i 0} {$i < [llength $::argv]} {incr i} {
    set arg [lindex $::argv $i]
    switch -- $arg {
        "--run"          { set run_flow 1 }
        "--skip-ip"      { set create_ip 0 }
        "--skip-bd"      { set create_bd 0 }
        "--project-name" { incr i; set project_name [lindex $::argv $i] }
        "--project-dir"  { incr i; set project_dir [file normalize [lindex $::argv $i]] }
        "--part"         { incr i; set part_name [lindex $::argv $i] }
        "--board-part"   { incr i; set board_part [lindex $::argv $i] }
        "--top"          { incr i; set top_name [lindex $::argv $i] }
        "--bd-name"      { incr i; set bd_name [lindex $::argv $i] }
        "--ip-repo-dir"  { incr i; set ip_repo_dir [file normalize [lindex $::argv $i]] }
        "--jobs"         { incr i; set jobs [lindex $::argv $i] }
        "--ip-clk-freq-hz" { incr i; set ip_clk_freq_hz [lindex $::argv $i] }
        "--ignore-freq-hz" { set ignore_freq_hz_compat 1 }
        "--strict-freq-hz" { set ignore_freq_hz_compat 0 }
        "--help"         { print_help; return }
        default           { puts "WARNING: unknown argument ignored: $arg" }
    }
}

if {$project_dir eq ""} { set project_dir [file normalize [file join $repo_root "vivado" "build" $project_name]] }
if {$ip_repo_dir eq ""} { set ip_repo_dir [file normalize [file join $repo_root "vivado" "ip_repo"]] }

set rtl_dir [file normalize [file join $repo_root "src" "rtl" "generated" "src_ultra_sequential"]]

puts "INFO: repo_root    = $repo_root"
puts "INFO: project_name = $project_name"
puts "INFO: project_dir  = $project_dir"
puts "INFO: part_name    = $part_name"
puts "INFO: board_part   = $board_part"
puts "INFO: top          = $top_name"
puts "INFO: rtl_dir      = $rtl_dir"
puts "INFO: ip_repo_dir  = $ip_repo_dir"
puts "INFO: bd_name      = $bd_name"
puts "INFO: ip_clk_freq = $ip_clk_freq_hz"
puts "INFO: ignore_freq_hz_compat = $ignore_freq_hz_compat"

if {[file exists $project_dir]} {
    file delete -force $project_dir
}
file mkdir $project_dir

create_project -force $project_name $project_dir -part $part_name
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]
set_property default_lib xil_defaultlib [current_project]
set_property source_mgmt_mode All [current_project]

if {$board_part ne ""} {
    if {[catch {set_property board_part $board_part [current_project]} msg]} {
        puts "WARNING: board_part could not be set: $msg"
    }
}

if {[get_filesets -quiet sources_1] eq ""} { create_fileset -srcset sources_1 }
if {[get_filesets -quiet sim_1] eq ""}     { create_fileset -simset sim_1 }
if {[get_filesets -quiet constrs_1] eq ""} { create_fileset -constrset constrs_1 }

set_property include_dirs [list $rtl_dir] [get_filesets sources_1]
set_property include_dirs [list $rtl_dir] [get_filesets sim_1]

set rtl_files [list \
    [file join $rtl_dir "scales.vh"] \
    [file join $rtl_dir "axi_input.sv"] \
    [file join $rtl_dir "axi_output.sv"] \
    [file join $rtl_dir "dsp.sv"] \
    [file join $rtl_dir "mem_a.sv"] \
    [file join $rtl_dir "mem_b.sv"] \
    [file join $rtl_dir "mlp_sequential.sv"] \
    [file join $rtl_dir "mlp_sequential_ctrl.sv"] \
    [file join $rtl_dir "mlp_sequential_datapath.sv"] \
    [file join $rtl_dir "rom_act_all.sv"] \
    [file join $rtl_dir "rom_bias_all.sv"] \
    [file join $rtl_dir "rom_weight_all.sv"] \
    [file join $rtl_dir "soc_mlp.sv"] \
]
add_existing_files sources_1 $rtl_files "SystemVerilog" 1

set vh_path [file normalize [file join $rtl_dir "scales.vh"]]
if {[llength [get_files -quiet $vh_path]] > 0} {
    set_property file_type {Verilog Header} [get_files $vh_path]
}

foreach f [get_files -of_objects [get_filesets sources_1]] {
    set fname [get_property NAME $f]
    if {[string match "*.sv" $fname]} {
        set_property file_type {SystemVerilog} $f
    }
}

set_property top $top_name [get_filesets sources_1]
update_compile_order -fileset sources_1

# -----------------------------------------------------------------------------
# IP packaging and block-design generation
# -----------------------------------------------------------------------------
proc ipx_add_axis_interface {core if_name if_mode prefix} {
    if {[llength [ipx::get_bus_interfaces $if_name -of_objects $core -quiet]] == 0} {
        ipx::add_bus_interface $if_name $core
    }
    set bus_if [ipx::get_bus_interfaces $if_name -of_objects $core]
    set_property bus_type_vlnv xilinx.com:interface:axis:1.0 $bus_if
    set_property abstraction_type_vlnv xilinx.com:interface:axis_rtl:1.0 $bus_if
    set_property interface_mode $if_mode $bus_if

    foreach {logical physical} [list \
        TDATA  ${prefix}_tdata \
        TKEEP  ${prefix}_tkeep \
        TLAST  ${prefix}_tlast \
        TVALID ${prefix}_tvalid \
        TREADY ${prefix}_tready \
    ] {
        if {[llength [ipx::get_ports $physical -of_objects $core -quiet]] == 0} {
            puts "WARNING: IP port $physical not found while mapping $if_name/$logical"
            continue
        }
        if {[llength [ipx::get_port_maps $logical -of_objects $bus_if -quiet]] == 0} {
            ipx::add_port_map $logical $bus_if
        }
        set_property physical_name $physical [ipx::get_port_maps $logical -of_objects $bus_if]
    }
}

proc ipx_set_bus_parameter_value {bus_if param_name param_value} {
    if {$bus_if eq ""} { return }
    set params [ipx::get_bus_parameters $param_name -of_objects $bus_if -quiet]
    if {[llength $params] == 0} {
        catch {ipx::add_bus_parameter $param_name $bus_if}
        set params [ipx::get_bus_parameters $param_name -of_objects $bus_if -quiet]
    }
    if {[llength $params]} {
        catch {set_property value $param_value $params}
    }
}


proc try_set_property_if_exists {obj prop value} {
    if {$obj eq ""} { return 0 }
    if {[catch {set props [list_property $obj]}]} { return 0 }
    if {[lsearch -exact $props $prop] >= 0} {
        if {[catch {set_property $prop $value $obj} msg]} {
            puts "INFO: cannot set $prop on $obj to $value: $msg"
            return 0
        }
        puts "INFO: set $prop=$value on $obj"
        return 1
    }
    return 0
}

proc ipx_try_ignore_freq_hz_compat {bus_if} {
    if {$bus_if eq ""} { return }

    # Vivado versions expose the IP-packager "Ignore FREQ_HZ" option with
    # different internal property names. Try the known spellings and keep this
    # non-fatal so the script remains portable across Vivado releases.
    foreach p [ipx::get_bus_parameters FREQ_HZ -of_objects $bus_if -quiet] {
        foreach {prop val} [list \
            IS_IGNORED true \
            IS_IGNORE true \
            IGNORE true \
            IGNORE_PARAM true \
            IGNORE_IN_BD true \
            COMPATIBILITY_IGNORE true \
            VALUE_RESOLVE_TYPE user \
            value_resolve_type user \
        ] {
            try_set_property_if_exists $p $prop $val
        }
    }

    foreach {prop val} [list \
        CONFIG.FREQ_HZ.IGNORE true \
        CONFIG.FREQ_HZ_IGNORE true \
        FREQ_HZ_IGNORE true \
        IGNORE_FREQ_HZ true \
        COMPATIBILITY.FREQ_HZ.IGNORE true \
        COMPATIBILITY_IGNORE_FREQ_HZ true \
    ] {
        try_set_property_if_exists $bus_if $prop $val
    }
}

proc bd_try_ignore_freq_hz_compat {obj_name} {
    foreach obj [concat [get_bd_pins -quiet $obj_name] [get_bd_intf_pins -quiet $obj_name]] {
        foreach {prop val} [list \
            CONFIG.FREQ_HZ.IGNORE true \
            CONFIG.FREQ_HZ_IGNORE true \
            FREQ_HZ_IGNORE true \
            IGNORE_FREQ_HZ true \
            COMPATIBILITY.FREQ_HZ.IGNORE true \
            COMPATIBILITY_IGNORE_FREQ_HZ true \
        ] {
            try_set_property_if_exists $obj $prop $val
        }
    }
}

proc set_bd_property_if_exists {obj prop value} {
    if {$obj eq ""} { return }
    set props [list_property $obj]
    if {[lsearch -exact $props $prop] >= 0} {
        set_property $prop $value $obj
    } else {
        puts "INFO: property $prop not present on $obj; skipping"
    }
}

proc package_soc_mlp_ip {ip_repo_dir ip_vendor ip_library ip_name ip_version top_name ip_clk_freq_hz ignore_freq_hz_compat} {
    set ip_core_dir [file normalize [file join $ip_repo_dir $ip_name]]
    file mkdir $ip_repo_dir
    if {[file exists $ip_core_dir]} {
        puts "INFO: removing previous packaged IP: $ip_core_dir"
        file delete -force $ip_core_dir
    }

    set_property top $top_name [get_filesets sources_1]
    update_compile_order -fileset sources_1

    puts "INFO: RTL top for IP packaging = [get_property top [get_filesets sources_1]]"
    puts "INFO: packaging $top_name as IP: $ip_vendor:$ip_library:$ip_name:$ip_version"

    ipx::package_project \
        -root_dir $ip_core_dir \
        -vendor $ip_vendor \
        -library $ip_library \
        -taxonomy /UserIP \
        -import_files \
        -force

    set core [ipx::current_core]
    set_property vendor $ip_vendor $core
    set_property library $ip_library $core
    set_property name $ip_name $core
    set_property version $ip_version $core
    set_property display_name $ip_name $core
    set_property description "Auto-packaged SOC_MLP Ultra96 sequential AXI IP generated from mem_layout.json." $core
    set_property supported_families {zynquplus Production} $core

    # AXI-Stream interfaces first, then associate both of them to clk.
    # If clk is configured before the stream interfaces exist, Vivado may keep
    # only one ASSOCIATED_BUSIF value during auto-inference.
    ipx_add_axis_interface $core s_axis_s2mm slave  s_axis_s2mm
    ipx_add_axis_interface $core m_axis_mm2s master m_axis_mm2s

    # The generated Ultra96 top exposes clk/rstn.
    catch {ipx::infer_bus_interface clk xilinx.com:signal:clock_rtl:1.0 $core}
    catch {ipx::infer_bus_interface rstn xilinx.com:signal:reset_rtl:1.0 $core}

    set clk_if [ipx::get_bus_interfaces clk -of_objects $core -quiet]
    if {$clk_if ne ""} {
        ipx_set_bus_parameter_value $clk_if ASSOCIATED_BUSIF "s_axis_s2mm:m_axis_mm2s"
        ipx_set_bus_parameter_value $clk_if ASSOCIATED_RESET "rstn"
        # Match the PS/DMA clock used by Ultra96/Vivado. On this setup Vivado
        # reports pl_clk0 as 99999001 Hz, not exactly 100000000 Hz. If the IP is
        # packaged with 100000000, IP Integrator treats it as fixed metadata and
        # validate_bd_design fails on FREQ_HZ mismatches.
        ipx_set_bus_parameter_value $clk_if FREQ_HZ $ip_clk_freq_hz
    }

    set rst_if [ipx::get_bus_interfaces rstn -of_objects $core -quiet]
    if {$rst_if ne ""} {
        ipx_set_bus_parameter_value $rst_if POLARITY ACTIVE_LOW
    }

    foreach if_name [list s_axis_s2mm m_axis_mm2s] {
        set axis_if [ipx::get_bus_interfaces $if_name -of_objects $core -quiet]
        if {$axis_if ne ""} {
            ipx_set_bus_parameter_value $axis_if FREQ_HZ $ip_clk_freq_hz
        }
    }

    if {$ignore_freq_hz_compat} {
        foreach if_name [list clk s_axis_s2mm m_axis_mm2s] {
            set if_obj [ipx::get_bus_interfaces $if_name -of_objects $core -quiet]
            if {$if_obj ne ""} { ipx_try_ignore_freq_hz_compat $if_obj }
        }
    }

    ipx::check_integrity $core
    ipx::save_core $core
    ipx::unload_core $core

    set component_xml [file join $ip_core_dir component.xml]
    if {![file exists $component_xml]} {
        error "IP packaging failed: component.xml was not generated at $component_xml"
    }

    set_property ip_repo_paths [list $ip_repo_dir] [current_project]
    update_ip_catalog -rebuild

    puts "INFO: packaged IP available in: $ip_core_dir/component.xml"
    return [file join $ip_core_dir component.xml]
}

proc maybe_connect_bd_intf {from_pin to_pin net_name} {
    if {[llength [get_bd_intf_pins -quiet $from_pin]] && [llength [get_bd_intf_pins -quiet $to_pin]]} {
        connect_bd_intf_net -intf_net $net_name [get_bd_intf_pins $from_pin] [get_bd_intf_pins $to_pin]
    } else {
        puts "WARNING: cannot connect interface $from_pin -> $to_pin"
    }
}

proc maybe_connect_bd_net {net_name pin_list} {
    set pins [list]
    foreach p $pin_list {
        if {[llength [get_bd_pins -quiet $p]]} {
            lappend pins [get_bd_pins $p]
        } else {
            puts "WARNING: missing BD pin $p"
        }
    }
    if {[llength $pins] >= 2} {
        connect_bd_net -net $net_name {*}$pins
    }
}

proc get_bd_pin_freq_hz {pin_name fallback_hz} {
    set pin [get_bd_pins -quiet $pin_name]
    if {[llength $pin]} {
        foreach prop {CONFIG.FREQ_HZ FREQ_HZ} {
            set value ""
            catch {set value [get_property $prop $pin]}
            if {$value ne ""} { return $value }
        }
    }
    return $fallback_hz
}

proc set_bd_obj_freq_hz {obj_name freq_hz} {
    set pins [get_bd_pins -quiet $obj_name]
    if {[llength $pins]} {
        catch {set_property CONFIG.FREQ_HZ $freq_hz $pins}
        catch {set_property FREQ_HZ $freq_hz $pins}
    }

    set intfs [get_bd_intf_pins -quiet $obj_name]
    if {[llength $intfs]} {
        catch {set_property CONFIG.FREQ_HZ $freq_hz $intfs}
        catch {set_property FREQ_HZ $freq_hz $intfs}
    }
}

proc align_soc_mlp_axis_metadata {freq_hz ignore_freq_hz_compat} {
    puts "INFO: aligning SOC_MLP/DMA AXI-Stream metadata to FREQ_HZ=$freq_hz"

    foreach p [list \
        SOC_MLP_0/clk \
        SOC_MLP_0/s_axis_s2mm \
        SOC_MLP_0/m_axis_mm2s \
        axi_dma_0/M_AXIS_MM2S \
        axi_dma_0/S_AXIS_S2MM \
    ] {
        set_bd_obj_freq_hz $p $freq_hz
    }

    # Keep the IP and DMA AXI-Stream metadata coherent with the 16-bit stream.
    # Some packaged SystemVerilog IPs do not accept cell parameter propagation,
    # so set the interface-pin properties directly and ignore unsupported forms.
    foreach intf [list \
        SOC_MLP_0/s_axis_s2mm \
        SOC_MLP_0/m_axis_mm2s \
        axi_dma_0/M_AXIS_MM2S \
        axi_dma_0/S_AXIS_S2MM \
    ] {
        set ip [get_bd_intf_pins -quiet $intf]
        if {[llength $ip]} {
            catch {set_property CONFIG.TDATA_NUM_BYTES 2 $ip}
            catch {set_property TDATA_NUM_BYTES 2 $ip}
        }
    }

    if {$ignore_freq_hz_compat} {
        foreach p [list             SOC_MLP_0/clk             SOC_MLP_0/s_axis_s2mm             SOC_MLP_0/m_axis_mm2s             axi_dma_0/M_AXIS_MM2S             axi_dma_0/S_AXIS_S2MM         ] {
            bd_try_ignore_freq_hz_compat $p
        }
    }

    # If the user parameters are visible on the packaged IP, force the same
    # 16-bit stream width as the AXI DMA. Unsupported parameter names are ignored.
    set soc [get_bd_cells -quiet SOC_MLP_0]
    if {[llength $soc]} {
        foreach {cfg val} [list \
            CONFIG.AXIS_DATA_W {16} \
            CONFIG.AXIS_KEEP_W {2} \
        ] {
            catch {set_property $cfg $val $soc}
        }
    }
}

proc create_and_set_bd_wrapper {bd_name} {
    set bd_file [get_files ${bd_name}.bd]
    if {[llength $bd_file] == 0} {
        error "Cannot create wrapper: BD file not found for $bd_name"
    }

    generate_target all $bd_file

    set wrapper_path [make_wrapper -files $bd_file -top]
    if {[llength $wrapper_path] == 0} {
        set wrapper_path [file join [get_property DIRECTORY [current_project]] \
            [get_property NAME [current_project]].srcs sources_1 bd $bd_name hdl ${bd_name}_wrapper.v]
    }

    if {![file exists $wrapper_path]} {
        error "Wrapper was not generated: $wrapper_path"
    }

    add_files -norecurse -fileset sources_1 $wrapper_path
    set_property top ${bd_name}_wrapper [get_filesets sources_1]
    update_compile_order -fileset sources_1

    puts "INFO: wrapper generated: $wrapper_path"
    puts "INFO: synthesis top set to ${bd_name}_wrapper"
}

proc create_soc_mlp_block_design {bd_name ip_vlnv ip_clk_freq_hz ignore_freq_hz_compat} {
    if {[llength [get_bd_designs -quiet $bd_name]]} {
        puts "INFO: block design $bd_name already exists. Recreating it."
        close_bd_design $bd_name
        file delete -force [file join [get_property DIRECTORY [current_project]] [get_property NAME [current_project]].srcs sources_1 bd $bd_name]
    }

    create_bd_design $bd_name
    current_bd_design $bd_name

    set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ultra_ps_e_0]

    # Do not blindly apply the board preset when board_part is empty.
    # With Vivado 2025.2 this can enable unwanted PS masters.
    # The sequential DMA design uses only:
    #   - M_AXI_HPM0_FPD for AXI-Lite DMA control
    #   - S_AXI_HP0_FPD  for DMA DDR traffic
    # M_AXI_HPM0_LPD is intentionally disabled and must not appear.
    set current_board_part ""
    catch {set current_board_part [get_property board_part [current_project]]}
    catch {set current_board_part [get_property BOARD_PART [current_project]]}
    if {$current_board_part ne ""} {
        catch {apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps} board_msg
        if {[info exists board_msg] && $board_msg ne ""} { puts "INFO: PS board automation: $board_msg" }
    } else {
        puts "INFO: board_part is empty; skipping PS board automation preset."
    }

    set_property -dict [list \
        CONFIG.PSU__FPGA_PL0_ENABLE {1} \
        CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} \
        CONFIG.PSU__USE__M_AXI_GP0 {1} \
        CONFIG.PSU__USE__M_AXI_GP1 {0} \
        CONFIG.PSU__USE__M_AXI_GP2 {0} \
        CONFIG.PSU__USE__S_AXI_GP2 {1} \
        CONFIG.PSU__USE__IRQ0 {1} \
        CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    ] $ps

    # Extra guards for ZynqMP PS presets: disable unused HPM masters.
    # CONFIG.PSU__USE__M_AXI_GP2 is the important one here: on ZynqMP it
    # corresponds to the unwanted M_AXI_HPM0_LPD path in many Vivado versions.
    # The other names are kept as compatibility aliases when the property exists.
    foreach {cfg val} [list \
        CONFIG.PSU__USE__M_AXI_GP1 {0} \
        CONFIG.PSU__USE__M_AXI_GP2 {0} \
        CONFIG.PSU__USE__M_AXI_HPM1_FPD {0} \
        CONFIG.PSU__USE__M_AXI_HPM0_LPD {0} \
        CONFIG.PSU__USE__M_AXI_LPD {0} \
    ] {
        set_bd_property_if_exists $ps $cfg $val
    }

    # If this pin still exists, Vivado kept the unused LPD master alive.
    # Do not connect its clock as a workaround: fail early with a clear message.
    if {[llength [get_bd_intf_pins -quiet zynq_ultra_ps_e_0/M_AXI_HPM0_LPD]]} {
        puts "WARNING: M_AXI_HPM0_LPD is still present after PS configuration."
        puts "WARNING: The design does not use it; check the PS property name for this Vivado version."
    }

    set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0]
    set_property -dict [list \
        CONFIG.c_include_sg {0} \
        CONFIG.c_include_mm2s {1} \
        CONFIG.c_include_s2mm {1} \
        CONFIG.c_addr_width {40} \
        CONFIG.c_s_axis_s2mm_tdata_width {16} \
        CONFIG.c_m_axis_mm2s_tdata_width {16} \
        CONFIG.c_m_axi_mm2s_data_width {32} \
        CONFIG.c_m_axi_s2mm_data_width {32} \
        CONFIG.c_sg_length_width {23} \
    ] $dma

    set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0]

    set irq_concat [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0]
    set_property CONFIG.NUM_PORTS {2} $irq_concat

    set sc_hp [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_0]
    set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $sc_hp

    set sc_lite [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_1]
    set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $sc_lite

    set mlp [create_bd_cell -type ip -vlnv $ip_vlnv SOC_MLP_0]

    # AXI-Stream data path:
    # DDR -> DMA MM2S -> SOC_MLP -> DMA S2MM -> DDR.
    maybe_connect_bd_intf axi_dma_0/M_AXIS_MM2S SOC_MLP_0/s_axis_s2mm axi_dma_mm2s_to_soc_mlp
    maybe_connect_bd_intf SOC_MLP_0/m_axis_mm2s axi_dma_0/S_AXIS_S2MM soc_mlp_to_axi_dma_s2mm

    # AXI memory-mapped data path.
    maybe_connect_bd_intf axi_dma_0/M_AXI_MM2S smartconnect_0/S00_AXI dma_mm2s_to_hp_sc
    maybe_connect_bd_intf axi_dma_0/M_AXI_S2MM smartconnect_0/S01_AXI dma_s2mm_to_hp_sc
    maybe_connect_bd_intf smartconnect_0/M00_AXI zynq_ultra_ps_e_0/S_AXI_HP0_FPD hp_sc_to_ps_hp0

    # AXI-Lite control path.
    maybe_connect_bd_intf zynq_ultra_ps_e_0/M_AXI_HPM0_FPD smartconnect_1/S00_AXI ps_hpm0_to_lite_sc
    maybe_connect_bd_intf smartconnect_1/M00_AXI axi_dma_0/S_AXI_LITE lite_sc_to_dma_lite

    # Clock/reset. SOC_MLP generated top uses clk/rstn.
    maybe_connect_bd_net pl_clk0 [list \
        zynq_ultra_ps_e_0/pl_clk0 \
        proc_sys_reset_0/slowest_sync_clk \
        axi_dma_0/s_axi_lite_aclk \
        axi_dma_0/m_axi_mm2s_aclk \
        axi_dma_0/m_axi_s2mm_aclk \
        smartconnect_0/aclk \
        smartconnect_1/aclk \
        zynq_ultra_ps_e_0/saxihp0_fpd_aclk \
        zynq_ultra_ps_e_0/maxihpm0_fpd_aclk \
        SOC_MLP_0/clk \
    ]

    maybe_connect_bd_net pl_resetn0 [list \
        zynq_ultra_ps_e_0/pl_resetn0 \
        proc_sys_reset_0/ext_reset_in \
    ]

    maybe_connect_bd_net peripheral_aresetn [list \
        proc_sys_reset_0/peripheral_aresetn \
        axi_dma_0/axi_resetn \
        smartconnect_0/aresetn \
        smartconnect_1/aresetn \
        SOC_MLP_0/rstn \
    ]

    maybe_connect_bd_net dma_irq_mm2s [list axi_dma_0/mm2s_introut xlconcat_0/In0]
    maybe_connect_bd_net dma_irq_s2mm [list axi_dma_0/s2mm_introut xlconcat_0/In1]
    maybe_connect_bd_net ps_irq0 [list xlconcat_0/dout zynq_ultra_ps_e_0/pl_ps_irq0]

    catch {assign_bd_address -offset 0xA0000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force} msg
    if {[info exists msg] && $msg ne ""} { puts "INFO: AXI-Lite address assignment: $msg" }

    catch {assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_DDR_LOW] -force} msg
    if {[info exists msg] && $msg ne ""} { puts "INFO: MM2S address assignment: $msg" }

    catch {assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_DDR_LOW] -force} msg
    if {[info exists msg] && $msg ne ""} { puts "INFO: S2MM address assignment: $msg" }

    # Vivado may report PS pl_clk0 as 99999001 Hz, not exactly 100000000 Hz.
    # Use the actual BD clock property everywhere on the stream path before validation.
    set pl_clk_freq [get_bd_pin_freq_hz zynq_ultra_ps_e_0/pl_clk0 $ip_clk_freq_hz]
    align_soc_mlp_axis_metadata $pl_clk_freq $ignore_freq_hz_compat

    validate_bd_design
    save_bd_design

    create_and_set_bd_wrapper $bd_name

    puts "INFO: block design generated: $bd_name"
}

if {$create_ip} {
    set component_xml [package_soc_mlp_ip $ip_repo_dir $ip_vendor $ip_library $ip_name $ip_version $top_name $ip_clk_freq_hz $ignore_freq_hz_compat]
} else {
    puts "WARNING: IP packaging skipped by --skip-ip. BD creation requires an existing $ip_vendor:$ip_library:$ip_name:$ip_version IP."
    set_property ip_repo_paths [list $ip_repo_dir] [current_project]
    update_ip_catalog -rebuild
}

if {$create_bd} {
    create_soc_mlp_block_design $bd_name "$ip_vendor:$ip_library:$ip_name:$ip_version" $ip_clk_freq_hz $ignore_freq_hz_compat
} else {
    puts "WARNING: block design generation skipped by --skip-bd. Synthesis top remains $top_name."
}

# Ensure default runs exist.
if {[get_runs -quiet synth_1] eq ""} {
    set synth_flow [lindex [get_flows "Vivado Synthesis*"] 0]
    if {$synth_flow ne ""} {
        create_run -name synth_1 -flow $synth_flow -strategy "Vivado Synthesis Defaults" -constrset constrs_1
    }
}
if {[get_runs -quiet impl_1] eq "" && [get_runs -quiet synth_1] ne ""} {
    set impl_flow [lindex [get_flows "Vivado Implementation*"] 0]
    if {$impl_flow ne ""} {
        create_run -name impl_1 -parent_run synth_1 -flow $impl_flow -strategy "Vivado Implementation Defaults" -constrset constrs_1
    }
}
if {[get_runs -quiet synth_1] ne ""} { current_run -synthesis [get_runs synth_1] }
if {[get_runs -quiet impl_1] ne ""}  { current_run -implementation [get_runs impl_1] }

if {$run_flow} {
    puts "INFO: Running synthesis/implementation/bitstream"
    launch_runs synth_1 -jobs $jobs
    wait_on_run synth_1
    launch_runs impl_1 -to_step write_bitstream -jobs $jobs
    wait_on_run impl_1
}

puts ""
puts "INFO: Vivado project generated successfully."
puts "INFO: Project: $project_dir/$project_name.xpr"
puts "INFO: Topology: 6 -> 66 -> 50 -> 38 -> 28 -> 20 -> 2"
puts "INFO: BD: $bd_name"
puts "INFO: Next: open_project $project_dir/$project_name.xpr"
