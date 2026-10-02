// Live bridge between APDM Opals (via the access point) and Python.
//
// Uses APDM's own Java SDK (apdm.jar + libapdm.dylib, both shipped inside
// Motion Studio). libapdm.dylib is x86_64 only, so run this with Motion
// Studio's bundled x86_64 JRE (Rosetta on Apple silicon) -- see
// src/stride_imu/live/scripts/build_apdm_bridge.sh and stride_imu.live.sources.bridge_command.
//
// Modes:
//   probe                       load the native library, try to open the AP, report
//   configure <rate_hz> <sd 0|1> Opals DOCKED + AP plugged in: configure for
//                               wireless streaming (sd=1 also logs to the Opal
//                               flash = "robust streaming"; lost packets are recoverable)
//   stream [max_latency_ms]     Opals undocked: stream to stdout (line protocol below)
//
// stdout line protocol (parsed by stride_imu.live.sources.BridgeSource):
//   M {"devices": {"<id>": "<label>", ...}, "rate": <hz>}
//   S <dev> <t_us> ax ay az gx gy gz mx my mz <button>
//   B <dev> <sync> <event_code> <text>
//   X <ap> <sync> <pin> <value>
//   I <info text>     E <error text>
// stdin commands:  "gpio <ap_index> <value>"  (AP GPIO_0 / sync-box out),  "quit"
import com.apdm.APDMException;
import com.apdm.Context;
import com.apdm.RecordRaw;
import com.apdm.swig.apdm_button_data_t;
import com.apdm.swig.apdm_external_sync_data_t;
import com.apdm.swig.apdm_record_t;
import com.apdm.swig.apdm_streaming_config_t;
import com.apdm.swig.button_event_data_t;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;

public class ApdmBridge {
    static PrintWriter out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out), 1 << 16));
    static final ConcurrentLinkedQueue<String> commands = new ConcurrentLinkedQueue<String>();
    static volatile boolean running = true;

    public static void main(String[] args) throws Exception {
        String mode = args.length > 0 ? args[0] : "probe";
        try {
            // the SDK logs every USB poll at DEBUG to stderr otherwise
            Context.setLoggingLevel(com.apdm.swig.apdm_logging_level_t.APDM_LL_WARNING);
            if (mode.equals("probe")) probe();
            else if (mode.equals("configure")) configure(Integer.parseInt(args[1]), args[2].equals("1"));
            else if (mode.equals("stream")) stream(args.length > 1 ? Integer.parseInt(args[1]) : 250);
            else { out.println("E unknown mode " + mode); }
        } catch (APDMException e) {
            out.println("E APDMException code=" + e.getCode() + " " + e.getMessage());
        } catch (Throwable t) {
            out.println("E " + t);
        }
        out.flush();
    }

    static void probe() throws Exception {
        out.println("I java " + System.getProperty("java.version") + " arch " + System.getProperty("os.arch"));
        Context ctx = Context.getInstance();       // loads libapdm (static init)
        out.println("I libapdm loaded");
        try {
            ctx.open();
            out.println("I access points configured: " + ctx.getNumberOfConfiguredAPs()
                    + ", devices: " + ctx.getNumberOfConfiguredDevices());
            out.println("I labels " + ctx.getLabelList());
            ctx.close();
        } catch (APDMException e) {
            out.println("E could not open an access point (plugged in? Motion Studio closed?): " + e);
        }
    }

    static void configure(int rateHz, boolean sd) throws Exception {
        Context ctx = Context.getInstance();
        ctx.open();
        apdm_streaming_config_t cfg = new apdm_streaming_config_t();
        com.apdm.swig.apdm.apdm_init_streaming_config(cfg);
        cfg.setEnable_accel(true);
        cfg.setEnable_gyro(true);
        cfg.setEnable_mag(true);
        cfg.setOutput_rate_hz(rateHz);
        cfg.setEnable_sd_card(sd);       // also log on the Opal = robust streaming
        cfg.setErase_sd_card(false);
        cfg.setButton_enable(true);      // Opal button presses -> button events
        ctx.autoConfigureDevicesAndAccessPointStreaming(cfg);
        out.println("I configured " + ctx.getNumberOfConfiguredDevices() + " Opals at " + rateHz
                + " Hz, sd logging " + sd + ". Undock them and wait for synchronized green blinking.");
        out.println("I labels " + ctx.getLabelList());
        ctx.close();
    }

    static void stream(int maxLatencyMs) throws Exception {
        Context ctx = Context.getInstance();
        ctx.open();
        ctx.setMaxLatency(maxLatencyMs);
        ctx.syncRecordHeadList();
        long n = ctx.getNumberOfConfiguredDevices();
        StringBuilder meta = new StringBuilder("M {\"devices\": {");
        List<String> labels = ctx.getLabelList();
        long rate = 0;
        for (int i = 0; i < n; i++) {
            long id = ctx.getDeviceIdByIndex(i);
            rate = ctx.getDeviceInfo(id).getSample_rate();
            if (i > 0) meta.append(", ");
            meta.append('"').append(id).append("\": \"").append(labels.get(i).replace("\"", "")).append('"');
        }
        meta.append("}, \"rate\": ").append(rate).append('}');
        out.println(meta);
        out.flush();

        Thread stdin = new Thread(new Runnable() {
            public void run() {
                try {
                    BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
                    String line;
                    while ((line = in.readLine()) != null) commands.add(line.trim());
                } catch (Exception ignored) { }
                commands.add("quit");
            }
        });
        stdin.setDaemon(true);
        stdin.start();

        while (running) {
            List<RecordRaw> list = ctx.getNextRecordList();
            for (RecordRaw rr : list) {
                apdm_record_t r = rr.record;
                out.print("S "); out.print(r.getDevice_info_serial_number());
                out.print(' '); out.print(r.getV2_sync_val64_us());
                out.print(' '); out.print(r.getAccl_x_axis_si());
                out.print(' '); out.print(r.getAccl_y_axis_si());
                out.print(' '); out.print(r.getAccl_z_axis_si());
                out.print(' '); out.print(r.getGyro_x_axis_si());
                out.print(' '); out.print(r.getGyro_y_axis_si());
                out.print(' '); out.print(r.getGyro_z_axis_si());
                out.print(' '); out.print(r.getMag_x_axis_si());
                out.print(' '); out.print(r.getMag_y_axis_si());
                out.print(' '); out.print(r.getMag_z_axis_si());
                out.print(' '); out.println(r.getButton_status());
            }
            apdm_button_data_t b;
            while ((b = ctx.getButtonEvent()) != null) {
                button_event_data_t d = b.getDevice_button_data();
                out.println("B " + d.getDevice_id() + " " + d.getEvent_sync_time() + " "
                        + d.getButton_event() + " " + d.getEvent_string());
            }
            apdm_external_sync_data_t x;
            while ((x = ctx.getSynchronizationEvent()) != null) {
                out.println("X " + x.getAp_id() + " " + x.getSync_value() + " " + x.getV2_pin() + " " + x.getData());
            }
            out.flush();
            String cmd;
            while ((cmd = commands.poll()) != null) handle(ctx, cmd);
            if (list.isEmpty()) Thread.sleep(2);
        }
        ctx.close();
    }

    static void handle(Context ctx, String cmd) {
        String[] p = cmd.split("\\s+");
        try {
            if (p[0].equals("quit")) running = false;
            else if (p[0].equals("gpio")) {
                ctx.setAPOutputGPIOValue(Long.parseLong(p[1]), Long.parseLong(p[2]));
                out.println("I gpio " + p[1] + " = " + p[2]);
            } else out.println("E unknown command " + cmd);
        } catch (Exception e) {
            out.println("E command '" + cmd + "' failed: " + e);
        }
    }
}
