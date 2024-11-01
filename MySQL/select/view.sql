create view uv_membertbl
as
	select memberId,memberAddress from membertbl;
    
select * from uv_membertbl;