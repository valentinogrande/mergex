use std::thread;

#[derive(Debug)]
struct Mergex<'a, T> {
    father: Option<Box<&'a Mergex<'a, T>>>,
    data: T,
    dirty: bool, // if true means that it is not sync with father node
}

unsafe impl<'a, T> Send for Mergex<'a, T> {}

impl<'a, T: Clone> Mergex<'a, T> {
    pub fn new(a: T) -> Self {
        Self {
            data: a,
            father: None,
            dirty: false,
        }
    }

    pub fn clone(&'a self) -> Self {
        // clone should be called when a new thread is spawned
        Self {
            data: self.data.clone(),
            father: Some(Box::new(self)),
            dirty: false,
        }
    }
}

#[test]
fn test() {
    let mergex = Mergex::new(7);

    {
        let t1 = mergex.clone();
        let t2 = mergex.clone();

        let a = thread::spawn(|| t1);
        let b = thread::spawn(|| t2);

        a.join();
        b.join();
    }

    dbg!(mergex);
}
