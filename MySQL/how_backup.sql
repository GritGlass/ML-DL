-- 백업 --
use shopDB;
SELECT * FROM producttbl;
-- Administration -> Data Export -> 백업 스키마 선택 -> 저장 위치 선택 -> 저장 --


-- producttbl 테이블만 삭제 --
DELETE FROM producttbl;


-- 복원 --
use sys;
-- Administration -> Data Import/Restore -> import from self-contrained file -> 스키마 선택 -> start import --