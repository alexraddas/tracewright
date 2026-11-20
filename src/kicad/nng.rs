//! Minimal NNG "REQ" client over NNG's IPC transport, enough to talk to KiCad's API server.
//!
//! Wire format (NNG SP protocol over IPC):
//! * On connect both peers send an 8-byte header `00 'S' 'P' 00 <proto u16 BE> 00 00`, where
//!   the protocol is the sender's own (REQ0 = 0x30, REP0 = 0x31).
//! * Each message is framed as one type byte (1) plus a u64 big-endian length, followed by the
//!   body. For REQ/REP the body starts with a 4-byte request id (high bit set) that the reply
//!   echoes back.
//! * On Windows an `ipc://PATH` URL is the named pipe `\\.\pipe\PATH`; elsewhere it is a Unix
//!   domain socket at PATH.

use anyhow::{Context, Result, anyhow, bail};
use std::io::{Read, Write};
use std::time::{Duration, Instant};

const REQ0: u16 = 0x30;
const REP0: u16 = 0x31;

trait Stream: Read + Write + Send {}
impl<T: Read + Write + Send> Stream for T {}

pub struct ReqSocket {
    stream: Box<dyn Stream>,
    next_id: u32,
}

fn open(url: &str, timeout: Duration) -> Result<Box<dyn Stream>> {
    let path = url.strip_prefix("ipc://").ok_or_else(|| anyhow!("unsupported socket URL {url} (expected ipc://)"))?;
    let start = Instant::now();
    loop {
        #[cfg(windows)]
        let r = std::fs::OpenOptions::new().read(true).write(true).open(format!(r"\\.\pipe\{path}")).map(|f| Box::new(f) as Box<dyn Stream>);
        #[cfg(unix)]
        let r = std::os::unix::net::UnixStream::connect(path).map(|s| Box::new(s) as Box<dyn Stream>);
        match r {
            Ok(s) => return Ok(s),
            // ERROR_PIPE_BUSY (231): all server instances busy; retry shortly.
            Err(e) if start.elapsed() < timeout && (e.raw_os_error() == Some(231) || e.kind() == std::io::ErrorKind::WouldBlock) => {
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => return Err(e).with_context(|| format!("cannot open KiCad API socket {url}")),
        }
    }
}

impl ReqSocket {
    pub fn connect(url: &str) -> Result<ReqSocket> {
        let mut stream = open(url, Duration::from_secs(5))?;
        let mut hdr = [0u8, b'S', b'P', 0, 0, 0, 0, 0];
        hdr[4..6].copy_from_slice(&REQ0.to_be_bytes());
        stream.write_all(&hdr).context("send SP header")?;
        stream.flush()?;
        let mut peer = [0u8; 8];
        stream.read_exact(&mut peer).context("read SP header")?;
        if peer[0..4] != [0, b'S', b'P', 0] {
            bail!("peer is not an NNG socket");
        }
        let proto = u16::from_be_bytes([peer[4], peer[5]]);
        if proto != REP0 {
            bail!("peer speaks NNG protocol {proto:#x}, expected REP0");
        }
        let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(1);
        Ok(ReqSocket { stream, next_id: seed & 0x7fff_ffff })
    }

    /// Send one request and wait for its reply.
    pub fn request(&mut self, payload: &[u8]) -> Result<Vec<u8>> {
        let id = 0x8000_0000 | (self.next_id & 0x7fff_ffff);
        self.next_id = self.next_id.wrapping_add(1);
        let len = (payload.len() + 4) as u64;
        let mut frame = Vec::with_capacity(13 + payload.len());
        frame.push(1u8);
        frame.extend_from_slice(&len.to_be_bytes());
        frame.extend_from_slice(&id.to_be_bytes());
        frame.extend_from_slice(payload);
        self.stream.write_all(&frame).context("send request")?;
        self.stream.flush()?;
        loop {
            let mut h = [0u8; 9];
            self.stream.read_exact(&mut h).context("read reply header")?;
            let n = u64::from_be_bytes(h[1..9].try_into().unwrap()) as usize;
            if n < 4 || n > (1usize << 31) {
                bail!("bad reply length {n}");
            }
            let mut body = vec![0u8; n];
            self.stream.read_exact(&mut body).context("read reply body")?;
            let rid = u32::from_be_bytes(body[0..4].try_into().unwrap());
            if rid == id {
                body.drain(0..4);
                return Ok(body);
            }
            // A reply to an earlier (abandoned) request: skip it.
        }
    }
}
